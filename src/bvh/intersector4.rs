use arrayvec::ArrayVec;
use wide::{CmpGe as _, CmpLe, CmpLt as _, f32x4};

use crate::{
    bvh::{simd_ray::SimdRay4x2Interleaved, *},
    ray::{Ray4, RayHit4},
    trace_stats::TraceStats,
};

pub struct BvhIntersector4 {}

impl BvhIntersector4 {
    fn intersect_node4(
        child_bbox: &Vec3x4,
        ray: &SimdRay4x2Interleaved,
    ) -> (f32x4, f32x4, f32x4, f32x4) {
        use safe_arch::*;

        let hit_shuffle = set_i32_m256i(7, 5, 3, 1, 6, 4, 2, 0);

        // R1   R1   R2   R2   R3   R3   R4   R4
        // MINL MINR MAXL MAXR MINL MINR MAXL MAXR
        let bb_min_max_x = load_m128_splat_m256(&m128::from(child_bbox.x.to_array()));
        let bb_min_max_y = load_m128_splat_m256(&m128::from(child_bbox.y.to_array()));
        let bb_min_max_z = load_m128_splat_m256(&m128::from(child_bbox.z.to_array()));

        // MINL MINR MINL MINR MINL MINR MINL MINR
        let bb_min_x = shuffle_m256::<0b01_00_01_00>(bb_min_max_x, bb_min_max_x);
        let bb_min_y = shuffle_m256::<0b01_00_01_00>(bb_min_max_y, bb_min_max_y);
        let bb_min_z = shuffle_m256::<0b01_00_01_00>(bb_min_max_z, bb_min_max_z);

        let bb_max_x = shuffle_m256::<0b11_10_11_10>(bb_min_max_x, bb_min_max_x);
        let bb_max_y = shuffle_m256::<0b11_10_11_10>(bb_min_max_y, bb_min_max_y);
        let bb_max_z = shuffle_m256::<0b11_10_11_10>(bb_min_max_z, bb_min_max_z);

        let tmin_x = fused_mul_sub_m256(bb_min_x, ray.dir_recip_x, ray.origin_dir_recip_x);
        let tmin_y = fused_mul_sub_m256(bb_min_y, ray.dir_recip_y, ray.origin_dir_recip_y);
        let tmin_z = fused_mul_sub_m256(bb_min_z, ray.dir_recip_z, ray.origin_dir_recip_z);
        let tmax_x = fused_mul_sub_m256(bb_max_x, ray.dir_recip_x, ray.origin_dir_recip_x);
        let tmax_y = fused_mul_sub_m256(bb_max_y, ray.dir_recip_y, ray.origin_dir_recip_y);
        let tmax_z = fused_mul_sub_m256(bb_max_z, ray.dir_recip_z, ray.origin_dir_recip_z);

        // R1    R1    R2    R2    R3    R3    R4    R4
        // NEARL NEARR NEARL NEARR NEARL NEARR NEARL NEARR
        let tnear = max_m256(
            max_m256(min_m256(tmin_x, tmax_x), min_m256(tmin_y, tmax_y)),
            min_m256(tmin_z, tmax_z),
        );

        let tfar = min_m256(
            min_m256(max_m256(tmin_x, tmax_x), max_m256(tmin_y, tmax_y)),
            max_m256(tmin_z, tmax_z),
        );

        let hit = shuffle_av_f32_all_m256(
            cmp_op_mask_m256::<{ cmp_op!(LessEqualOrdered) }>(
                max_m256(tnear, ray.near),
                min_m256(tfar, ray.far),
            ),
            hit_shuffle,
        );

        let hit_l = f32x4::from(cast_to_m128_from_m256(hit).to_array());
        let hit_r = f32x4::from(extract_m128_from_m256::<1>(hit).to_array());

        let near = shuffle_av_f32_all_m256(tnear, hit_shuffle);
        let near_l = f32x4::from(cast_to_m128_from_m256(near).to_array());
        let near_r = f32x4::from(extract_m128_from_m256::<1>(near).to_array());

        (hit_l, hit_r, near_l, near_r)
    }

    #[must_use]
    pub fn occluded4(bvh: &Bvh, ray: &Ray4, ray_valid: u32, stats: &mut TraceStats) -> u32 {
        stats.trace_start(ray_valid.count_ones() as u64);

        let mut active = ray_valid;
        let mut stack = ArrayVec::<(&BvhNode, u32), 32>::new();
        let simd_ray = SimdRay4x2Interleaved::from(ray);

        let mut cur_node = Some((&bvh.root_node, active));
        while let Some((node, node_active)) = cur_node {
            if node_active & active == 0 {
                cur_node = stack.pop();
                continue;
            }

            let num_valid_rays = node_active.count_ones();
            match node {
                BvhNode::Inner {
                    child_bbox,
                    children,
                } => {
                    stats.inner_visit(num_valid_rays as u64);

                    let (hit_mask_l, hit_mask_r, _, _) =
                        Self::intersect_node4(child_bbox, &simd_ray);

                    let hit_mask_l = hit_mask_l.move_mask() as u32 & active;
                    let hit_mask_r = hit_mask_r.move_mask() as u32 & active;

                    if hit_mask_l != 0 {
                        cur_node = Some((&children[0], hit_mask_l));

                        if hit_mask_r != 0 {
                            stack.push((&children[1], hit_mask_r));
                        }
                    } else if hit_mask_r != 0 {
                        cur_node = Some((&children[1], hit_mask_r));
                    } else {
                        cur_node = stack.pop();
                    }
                }
                BvhNode::Leaf { triangles_range } => {
                    stats.leaf_visit(num_valid_rays as u64, triangles_range.len() as u64);
                    for obj_idx in triangles_range.clone() {
                        let obj = &bvh.triangles[obj_idx];
                        let hit = obj.intersect_simd(&ray.origin, &ray.direction);
                        let hit_mask =
                            (hit.cmp_ge(ray.near) & hit.cmp_lt(ray.far)).move_mask() as u32;
                        active &= !hit_mask;
                        if active == 0 {
                            return !active & ray_valid;
                        }
                    }

                    cur_node = stack.pop();
                }
            }
        }

        !active & ray_valid
    }

    pub fn intersect4(bvh: &Bvh, ray_hit: &mut RayHit4, ray_valid: u32, stats: &mut TraceStats) {
        stats.trace_start(ray_valid.count_ones() as u64);

        let valid = f32x4::new([
            if ray_valid & 1 << 0 != 0 { -0. } else { 0. },
            if ray_valid & 1 << 1 != 0 { -0. } else { 0. },
            if ray_valid & 1 << 2 != 0 { -0. } else { 0. },
            if ray_valid & 1 << 3 != 0 { -0. } else { 0. },
        ]);
        ray_hit.ray.near = valid.blend(ray_hit.ray.near, f32x4::splat(f32::INFINITY));
        ray_hit.ray.far = valid.blend(ray_hit.ray.far, f32x4::splat(f32::NEG_INFINITY));

        let mut stack = ArrayVec::<_, 32>::new();
        let mut simd_ray = SimdRay4x2Interleaved::from(&ray_hit.ray);

        let mut cur_node = Some((&bvh.root_node, ray_hit.ray.near));
        while let Some((node, node_near)) = cur_node {
            let node_ray_valid = node_near.cmp_le(ray_hit.ray.far);

            if node_ray_valid.none() {
                cur_node = stack.pop();
                continue;
            }

            let num_valid_rays = node_ray_valid.move_mask().count_ones();
            match node {
                BvhNode::Inner {
                    child_bbox,
                    children,
                } => {
                    stats.inner_visit(num_valid_rays as u64);

                    let mut hits = ArrayVec::<_, 2>::new();
                    let (hit_l, hit_r, near_l, near_r) =
                        Self::intersect_node4(child_bbox, &simd_ray);

                    if hit_l.any() {
                        let dist = hit_l.blend(near_l, f32x4::splat(f32::INFINITY));
                        hits.push((&*children[0], dist));
                    }

                    if hit_r.any() {
                        let dist = hit_r.blend(near_r, f32x4::splat(f32::INFINITY));
                        hits.push((&*children[1], dist));
                    }

                    if hits.len() >= 2 && hits[0].1.cmp_lt(hits[1].1).any() {
                        hits.swap(0, 1);
                    }

                    if let Some(hit) = hits.pop() {
                        cur_node = Some(hit);
                        if let Some(hit) = hits.pop() {
                            stack.push(hit);
                        }
                    } else {
                        cur_node = stack.pop();
                    }
                }
                BvhNode::Leaf { triangles_range } => {
                    stats.leaf_visit(num_valid_rays as u64, triangles_range.len() as u64);
                    for obj_idx in triangles_range.clone() {
                        let obj = &bvh.triangles[obj_idx];
                        let hit = obj.intersect_simd(&ray_hit.ray.origin, &ray_hit.ray.direction);
                        let hit_mask = (hit.cmp_ge(ray_hit.ray.near) & hit.cmp_lt(ray_hit.ray.far))
                            .move_mask() as u32;
                        let hit: [f32; 4] = hit.into();
                        let far = ray_hit.ray.far.as_array_mut();
                        for i in 0..4 {
                            if hit_mask & 1 << i != 0 {
                                far[i] = hit[i];
                                ray_hit.obj_idx[i] = Some(bvh.object_indices[obj_idx] as u32);
                            }
                        }
                        if hit_mask != 0 {
                            simd_ray.update_far(far);
                        }
                    }

                    cur_node = stack.pop();
                }
            }
        }
    }
}
