use arrayvec::ArrayVec;
use safe_arch::m256;
use wide::{CmpGe as _, CmpLt as _};

use crate::{bvh::*, ray::Ray4, trace_stats::TraceStats};

impl Bvh {
    #[allow(clippy::too_many_arguments)]
    fn intersect_node4(
        child_bbox: &Vec3x4,
        dir_recip_x: m256,
        dir_recip_y: m256,
        dir_recip_z: m256,
        origin_dir_recip_x: m256,
        origin_dir_recip_y: m256,
        origin_dir_recip_z: m256,
        ray_near: m256,
        ray_far: m256,
        ray_valid: u32,
    ) -> (u32, u32) {
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

        let bb_near_x = blend_varying_m256(bb_min_x, bb_max_x, dir_recip_x);
        let bb_near_y = blend_varying_m256(bb_min_y, bb_max_y, dir_recip_y);
        let bb_near_z = blend_varying_m256(bb_min_z, bb_max_z, dir_recip_z);
        let bb_far_x = blend_varying_m256(bb_max_x, bb_min_x, dir_recip_x);
        let bb_far_y = blend_varying_m256(bb_max_y, bb_min_y, dir_recip_y);
        let bb_far_z = blend_varying_m256(bb_max_z, bb_min_z, dir_recip_z);

        let tnear_x = fused_mul_sub_m256(bb_near_x, dir_recip_x, origin_dir_recip_x);
        let tnear_y = fused_mul_sub_m256(bb_near_y, dir_recip_y, origin_dir_recip_y);
        let tnear_z = fused_mul_sub_m256(bb_near_z, dir_recip_z, origin_dir_recip_z);

        // R1    R1    R2    R2    R3    R3    R4    R4
        // NEARL NEARR NEARL NEARR NEARL NEARR NEARL NEARR
        let tnear = max_m256(max_m256(tnear_x, tnear_y), max_m256(tnear_z, ray_near));

        let tfar_x = fused_mul_sub_m256(bb_far_x, dir_recip_x, origin_dir_recip_x);
        let tfar_y = fused_mul_sub_m256(bb_far_y, dir_recip_y, origin_dir_recip_y);
        let tfar_z = fused_mul_sub_m256(bb_far_z, dir_recip_z, origin_dir_recip_z);

        let tfar = min_m256(min_m256(tfar_x, tfar_y), min_m256(tfar_z, ray_far));

        let hit_mask = move_mask_m256(shuffle_av_f32_all_m256(
            cmp_op_mask_m256::<{ cmp_op!(LessEqualOrdered) }>(tnear, tfar),
            hit_shuffle,
        )) as u32;

        let hit_mask_l = hit_mask & ray_valid;
        let hit_mask_r = hit_mask >> 4 & ray_valid;

        (hit_mask_l, hit_mask_r)
    }

    #[must_use]
    pub fn occluded4(&self, ray: &Ray4, ray_valid: u32, stats: &mut TraceStats) -> u32 {
        use safe_arch::*;

        stats.trace_start(ray_valid.count_ones() as u64);

        let mut occluded = 0u32;
        let mut stack = ArrayVec::<(&BvhNode, u32), 32>::new();

        let origin_x = {
            let a = ray.origin.x.as_array_ref();
            set_m256(a[3], a[3], a[2], a[2], a[1], a[1], a[0], a[0])
        };
        let origin_y = {
            let a = ray.origin.y.as_array_ref();
            set_m256(a[3], a[3], a[2], a[2], a[1], a[1], a[0], a[0])
        };
        let origin_z = {
            let a = ray.origin.z.as_array_ref();
            set_m256(a[3], a[3], a[2], a[2], a[1], a[1], a[0], a[0])
        };
        let ray_near = {
            let a = ray.near.as_array_ref();
            set_m256(a[3], a[3], a[2], a[2], a[1], a[1], a[0], a[0])
        };
        let ray_far = {
            let a = ray.far.as_array_ref();
            set_m256(a[3], a[3], a[2], a[2], a[1], a[1], a[0], a[0])
        };

        let dir_x = {
            let a = ray.direction.x.as_array_ref();
            set_m256(a[3], a[3], a[2], a[2], a[1], a[1], a[0], a[0])
        };
        let dir_y = {
            let a = ray.direction.y.as_array_ref();
            set_m256(a[3], a[3], a[2], a[2], a[1], a[1], a[0], a[0])
        };
        let dir_z = {
            let a = ray.direction.z.as_array_ref();
            set_m256(a[3], a[3], a[2], a[2], a[1], a[1], a[0], a[0])
        };

        let one = set_splat_m256(1.);
        let dir_recip_x = one / dir_x;
        let dir_recip_y = one / dir_y;
        let dir_recip_z = one / dir_z;

        let origin_dir_recip_x = origin_x * dir_recip_x;
        let origin_dir_recip_y = origin_y * dir_recip_y;
        let origin_dir_recip_z = origin_z * dir_recip_z;

        let mut cur_node = Some((&self.root_node, ray_valid));
        while let Some((node, node_ray_valid)) = cur_node {
            let num_valid_rays = node_ray_valid.count_ones();
            match node {
                BvhNode::Inner {
                    child_bbox,
                    children,
                } => {
                    stats.inner_visit(num_valid_rays as u64);

                    let (hit_mask_l, hit_mask_r) = Self::intersect_node4(
                        child_bbox,
                        dir_recip_x,
                        dir_recip_y,
                        dir_recip_z,
                        origin_dir_recip_x,
                        origin_dir_recip_y,
                        origin_dir_recip_z,
                        ray_near,
                        ray_far,
                        node_ray_valid & !occluded,
                    );

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
                        let obj = &self.triangles[obj_idx];
                        let hit = obj.intersect_simd(&ray.origin, &ray.direction);
                        let hit_mask =
                            (hit.cmp_ge(ray.near) & hit.cmp_lt(ray.far)).move_mask() as u32;
                        occluded |= hit_mask & ray_valid;
                        if occluded == ray_valid {
                            return occluded;
                        }
                    }

                    cur_node = stack.pop();
                }
            }
        }

        occluded
    }

    #[must_use]
    pub fn intersect4(
        &self,
        ray: &mut Ray4,
        ray_valid: u32,
        stats: &mut TraceStats,
    ) -> [Option<u32>; 4] {
        use safe_arch::*;

        stats.trace_start(ray_valid.count_ones() as u64);

        let mut hit_obj = [None; 4];
        let mut stack = ArrayVec::<(&BvhNode, u32), 32>::new();

        let origin_x = {
            let a = ray.origin.x.as_array_ref();
            set_m256(a[3], a[3], a[2], a[2], a[1], a[1], a[0], a[0])
        };
        let origin_y = {
            let a = ray.origin.y.as_array_ref();
            set_m256(a[3], a[3], a[2], a[2], a[1], a[1], a[0], a[0])
        };
        let origin_z = {
            let a = ray.origin.z.as_array_ref();
            set_m256(a[3], a[3], a[2], a[2], a[1], a[1], a[0], a[0])
        };
        let ray_near = {
            let a = ray.near.as_array_ref();
            set_m256(a[3], a[3], a[2], a[2], a[1], a[1], a[0], a[0])
        };

        let dir_x = {
            let a = ray.direction.x.as_array_ref();
            set_m256(a[3], a[3], a[2], a[2], a[1], a[1], a[0], a[0])
        };
        let dir_y = {
            let a = ray.direction.y.as_array_ref();
            set_m256(a[3], a[3], a[2], a[2], a[1], a[1], a[0], a[0])
        };
        let dir_z = {
            let a = ray.direction.z.as_array_ref();
            set_m256(a[3], a[3], a[2], a[2], a[1], a[1], a[0], a[0])
        };

        let one = set_splat_m256(1.);
        let dir_recip_x = one / dir_x;
        let dir_recip_y = one / dir_y;
        let dir_recip_z = one / dir_z;

        let origin_dir_recip_x = origin_x * dir_recip_x;
        let origin_dir_recip_y = origin_y * dir_recip_y;
        let origin_dir_recip_z = origin_z * dir_recip_z;

        let mut cur_node = Some((&self.root_node, ray_valid));
        while let Some((node, node_ray_valid)) = cur_node {
            let num_valid_rays = node_ray_valid.count_ones();
            match node {
                BvhNode::Inner {
                    child_bbox,
                    children,
                } => {
                    stats.inner_visit(num_valid_rays as u64);

                    // Load inside the loop since we modify it in the intersection test
                    let ray_far = {
                        let a = ray.far.as_array_ref();
                        set_m256(a[3], a[3], a[2], a[2], a[1], a[1], a[0], a[0])
                    };

                    let (hit_mask_l, hit_mask_r) = Self::intersect_node4(
                        child_bbox,
                        dir_recip_x,
                        dir_recip_y,
                        dir_recip_z,
                        origin_dir_recip_x,
                        origin_dir_recip_y,
                        origin_dir_recip_z,
                        ray_near,
                        ray_far,
                        node_ray_valid,
                    );

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
                        let obj = &self.triangles[obj_idx];
                        let hit = obj.intersect_simd(&ray.origin, &ray.direction);
                        let hit_mask = (hit.cmp_ge(ray.near) & hit.cmp_lt(ray.far)).move_mask()
                            as u32
                            & node_ray_valid;
                        let hit: [f32; 4] = hit.into();
                        let far = ray.far.as_array_mut();
                        for i in 0..4 {
                            if hit_mask & 1 << i != 0 {
                                far[i] = hit[i];
                                hit_obj[i] = Some(self.object_indices[obj_idx] as u32);
                            }
                        }
                    }

                    cur_node = stack.pop();
                }
            }
        }

        hit_obj
    }
}
