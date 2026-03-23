use std::ops::Range;

use arrayvec::ArrayVec;

use crate::{bvh::*, ray::StreamRay, trace_stats::TraceStats};

type RayIdx = u32;

impl Bvh {
    pub fn intersect_stream(
        &self,
        rays: &mut [StreamRay],
        hit_objects: &mut [Option<u32>],
        stats: &mut TraceStats,
    ) {
        assert!(rays.len() <= RayIdx::MAX as usize);
        use safe_arch::*;

        stats.trace_start(rays.len() as u64);

        let mut stack = ArrayVec::<_, 64>::new();
        let ray_list_len = rays.len() * 32;
        let mut ray_list_vec = vec![0; ray_list_len * 3];
        let ray_lists = {
            let (rl1, rest) = ray_list_vec.split_at_mut(ray_list_len);
            let (rl2, rl3) = rest.split_at_mut(ray_list_len);
            [rl1, rl2, rl3]
        };
        let mut ray_list_sizes = [0; 3];

        for (i, item) in ray_lists[0].iter_mut().enumerate().take(rays.len()) {
            *item = i as RayIdx;
        }
        ray_list_sizes[0] = rays.len();
        stack.push((&self.root_node, 0, 0));

        while let Some((node, list_idx, start_idx)) = stack.pop() {
            let mut active_ray_idx = start_idx;
            let last_active_ray_idx = ray_list_sizes[list_idx];

            ray_list_sizes[list_idx] = start_idx;

            match node {
                BvhNode::Inner {
                    child_bbox,
                    children,
                } => {
                    stats.inner_visit((last_active_ray_idx - active_ray_idx) as u64);

                    let neg_mask = set_splat_m256(-0.);
                    let pos_neg_mask = m256::from([0., 0., -0., -0., 0., 0., -0., -0.]);
                    let bb_min_max_x = load_m128_splat_m256(&m128::from_array(child_bbox.x.into()));
                    let bb_min_max_y = load_m128_splat_m256(&m128::from_array(child_bbox.y.into()));
                    let bb_min_max_z = load_m128_splat_m256(&m128::from_array(child_bbox.z.into()));

                    let bb_max_min_x =
                        shuffle_m256::<0b01_00_11_10>(bb_min_max_x, bb_min_max_x) ^ pos_neg_mask;
                    let bb_max_min_y =
                        shuffle_m256::<0b01_00_11_10>(bb_min_max_y, bb_min_max_y) ^ pos_neg_mask;
                    let bb_max_min_z =
                        shuffle_m256::<0b01_00_11_10>(bb_min_max_z, bb_min_max_z) ^ pos_neg_mask;
                    let bb_min_max_x = bb_min_max_x ^ pos_neg_mask;
                    let bb_min_max_y = bb_min_max_y ^ pos_neg_mask;
                    let bb_min_max_z = bb_min_max_z ^ pos_neg_mask;

                    let ray_list_sizes_orig = ray_list_sizes;
                    while active_ray_idx < last_active_ray_idx {
                        let ray_idx_a = ray_lists[list_idx][active_ray_idx] as usize;
                        let ray_idx_b = if active_ray_idx + 1 < last_active_ray_idx {
                            ray_lists[list_idx][active_ray_idx + 1] as usize
                        } else {
                            ray_idx_a
                        };

                        let origin_xyz_far_ab = set_m128_m256(
                            m128::from_array(rays[ray_idx_b].origin_far.into()),
                            m128::from_array(rays[ray_idx_a].origin_far.into()),
                        );

                        let origin_xyz_far_ab_neg = origin_xyz_far_ab ^ neg_mask;

                        let dir_recip_xyz_near_ab = set_m128_m256(
                            m128::from_array(rays[ray_idx_b].direction_recip_near.into()),
                            m128::from_array(rays[ray_idx_a].direction_recip_near.into()),
                        );

                        let origin_x =
                            shuffle_m256::<0b00_00_00_00>(origin_xyz_far_ab, origin_xyz_far_ab_neg);
                        let origin_y =
                            shuffle_m256::<0b01_01_01_01>(origin_xyz_far_ab, origin_xyz_far_ab_neg);
                        let origin_z =
                            shuffle_m256::<0b10_10_10_10>(origin_xyz_far_ab, origin_xyz_far_ab_neg);

                        let dir_recip_x = shuffle_m256::<0b00_00_00_00>(
                            dir_recip_xyz_near_ab,
                            dir_recip_xyz_near_ab,
                        );
                        let dir_recip_y = shuffle_m256::<0b01_01_01_01>(
                            dir_recip_xyz_near_ab,
                            dir_recip_xyz_near_ab,
                        );
                        let dir_recip_z = shuffle_m256::<0b10_10_10_10>(
                            dir_recip_xyz_near_ab,
                            dir_recip_xyz_near_ab,
                        );

                        let ray_near_far = shuffle_m256::<0b11_11_11_11>(
                            dir_recip_xyz_near_ab,
                            origin_xyz_far_ab_neg,
                        );

                        let origin_dir_recip_x = origin_x * dir_recip_x;
                        let origin_dir_recip_y = origin_y * dir_recip_y;
                        let origin_dir_recip_z = origin_z * dir_recip_z;

                        let bb_near_far_x =
                            blend_varying_m256(bb_min_max_x, bb_max_min_x, dir_recip_x);
                        let bb_near_far_y =
                            blend_varying_m256(bb_min_max_y, bb_max_min_y, dir_recip_y);
                        let bb_near_far_z =
                            blend_varying_m256(bb_min_max_z, bb_max_min_z, dir_recip_z);

                        let tnear_far_x =
                            fused_mul_sub_m256(bb_near_far_x, dir_recip_x, origin_dir_recip_x);
                        let tnear_far_y =
                            fused_mul_sub_m256(bb_near_far_y, dir_recip_y, origin_dir_recip_y);
                        let tnear_far_z =
                            fused_mul_sub_m256(bb_near_far_z, dir_recip_z, origin_dir_recip_z);

                        // near 0, near 1, -far 0, -far 1
                        let tnear_far = max_m256(
                            max_m256(max_m256(tnear_far_x, tnear_far_y), tnear_far_z),
                            ray_near_far,
                        );

                        // near 0, near 1, near 0, near 0
                        // far 0, far 1, near 1, near 1
                        let mask =
                            move_mask_m256(cmp_op_mask_m256::<{ cmp_op!(LessEqualOrdered) }>(
                                shuffle_m256::<0b00_00_01_00>(tnear_far, tnear_far),
                                shuffle_m256::<0b01_01_11_10>(tnear_far ^ neg_mask, tnear_far),
                            ));

                        let left_hit_a = mask & 0b1;
                        let right_hit_a = (mask >> 1) & 0b1;
                        let left_first_a = (mask >> 2) & 0b1;

                        ray_lists[0][ray_list_sizes[0]] = ray_idx_a as RayIdx;
                        ray_lists[1][ray_list_sizes[1]] = ray_idx_a as RayIdx;
                        ray_lists[2][ray_list_sizes[2]] = ray_idx_a as RayIdx;
                        ray_list_sizes[0] += (left_hit_a & left_first_a) as usize;
                        ray_list_sizes[1] += right_hit_a as usize;
                        ray_list_sizes[2] += (left_hit_a & (left_first_a ^ 0b1)) as usize;

                        if ray_idx_a != ray_idx_b {
                            let left_hit_b = (mask >> 4) & 0b1;
                            let right_hit_b = (mask >> 5) & 0b1;
                            let left_first_b = (mask >> 6) & 0b1;

                            ray_lists[0][ray_list_sizes[0]] = ray_idx_b as RayIdx;
                            ray_lists[1][ray_list_sizes[1]] = ray_idx_b as RayIdx;
                            ray_lists[2][ray_list_sizes[2]] = ray_idx_b as RayIdx;
                            ray_list_sizes[0] += (left_hit_b & left_first_b) as usize;
                            ray_list_sizes[1] += right_hit_b as usize;
                            ray_list_sizes[2] += (left_hit_b & (left_first_b ^ 0b1)) as usize;
                        }

                        active_ray_idx += 2;
                    }

                    if ray_list_sizes[2] - ray_list_sizes_orig[2] > 0 {
                        stack.push((&children[0], 2, ray_list_sizes_orig[2]));
                    }

                    if ray_list_sizes[1] - ray_list_sizes_orig[1] > 0 {
                        stack.push((&children[1], 1, ray_list_sizes_orig[1]));
                    }

                    if ray_list_sizes[0] - ray_list_sizes_orig[0] > 0 {
                        stack.push((&children[0], 0, ray_list_sizes_orig[0]));
                    }
                }
                BvhNode::Leaf { triangles_range } => {
                    stats.leaf_visit(
                        (last_active_ray_idx - active_ray_idx) as u64,
                        triangles_range.len() as u64,
                    );

                    self.intersect_objs(
                        triangles_range.clone(),
                        &ray_lists[list_idx][active_ray_idx..last_active_ray_idx],
                        rays,
                        hit_objects,
                    );
                }
            }
        }
    }

    pub fn occluded_stream(
        &self,
        rays: &mut [StreamRay],
        occluded: &mut [bool],
        stats: &mut TraceStats,
    ) {
        assert!(rays.len() <= RayIdx::MAX as usize);
        use safe_arch::*;

        stats.trace_start(rays.len() as u64);

        let mut stack = ArrayVec::<_, 64>::new();
        let ray_list_len = rays.len() * 32;
        let mut ray_list_vec = vec![0; ray_list_len * 2];
        let ray_lists = {
            let (rl1, rl2) = ray_list_vec.split_at_mut(ray_list_len);
            [rl1, rl2]
        };
        let mut ray_list_sizes = [0; 2];

        for (i, item) in ray_lists[0].iter_mut().enumerate().take(rays.len()) {
            *item = i as RayIdx;
        }
        ray_list_sizes[0] = rays.len();
        stack.push((&self.root_node, 0, 0));

        while let Some((node, list_idx, start_idx)) = stack.pop() {
            let mut active_ray_idx = start_idx;
            let last_active_ray_idx = ray_list_sizes[list_idx];

            ray_list_sizes[list_idx] = start_idx;

            match node {
                BvhNode::Inner {
                    child_bbox,
                    children,
                } => {
                    stats.inner_visit((last_active_ray_idx - active_ray_idx) as u64);

                    let neg_mask = set_splat_m256(-0.);
                    let pos_neg_mask = m256::from([0., 0., -0., -0., 0., 0., -0., -0.]);
                    let bb_min_max_x = load_m128_splat_m256(&m128::from_array(child_bbox.x.into()));
                    let bb_min_max_y = load_m128_splat_m256(&m128::from_array(child_bbox.y.into()));
                    let bb_min_max_z = load_m128_splat_m256(&m128::from_array(child_bbox.z.into()));

                    let bb_max_min_x =
                        shuffle_m256::<0b01_00_11_10>(bb_min_max_x, bb_min_max_x) ^ pos_neg_mask;
                    let bb_max_min_y =
                        shuffle_m256::<0b01_00_11_10>(bb_min_max_y, bb_min_max_y) ^ pos_neg_mask;
                    let bb_max_min_z =
                        shuffle_m256::<0b01_00_11_10>(bb_min_max_z, bb_min_max_z) ^ pos_neg_mask;
                    let bb_min_max_x = bb_min_max_x ^ pos_neg_mask;
                    let bb_min_max_y = bb_min_max_y ^ pos_neg_mask;
                    let bb_min_max_z = bb_min_max_z ^ pos_neg_mask;

                    let ray_list_sizes_orig = ray_list_sizes;
                    while active_ray_idx < last_active_ray_idx {
                        let ray_idx_a = ray_lists[list_idx][active_ray_idx] as usize;
                        if occluded[ray_idx_a] {
                            active_ray_idx += 1;
                            continue;
                        }

                        let ray_idx_b = if active_ray_idx + 1 < last_active_ray_idx {
                            ray_lists[list_idx][active_ray_idx + 1] as usize
                        } else {
                            ray_idx_a
                        };

                        let origin_xyz_far_ab = set_m128_m256(
                            m128::from_array(rays[ray_idx_b].origin_far.into()),
                            m128::from_array(rays[ray_idx_a].origin_far.into()),
                        );

                        let origin_xyz_far_ab_neg = origin_xyz_far_ab ^ neg_mask;

                        let dir_recip_xyz_near_ab = set_m128_m256(
                            m128::from_array(rays[ray_idx_b].direction_recip_near.into()),
                            m128::from_array(rays[ray_idx_a].direction_recip_near.into()),
                        );

                        let origin_x =
                            shuffle_m256::<0b00_00_00_00>(origin_xyz_far_ab, origin_xyz_far_ab_neg);
                        let origin_y =
                            shuffle_m256::<0b01_01_01_01>(origin_xyz_far_ab, origin_xyz_far_ab_neg);
                        let origin_z =
                            shuffle_m256::<0b10_10_10_10>(origin_xyz_far_ab, origin_xyz_far_ab_neg);

                        let dir_recip_x = shuffle_m256::<0b00_00_00_00>(
                            dir_recip_xyz_near_ab,
                            dir_recip_xyz_near_ab,
                        );
                        let dir_recip_y = shuffle_m256::<0b01_01_01_01>(
                            dir_recip_xyz_near_ab,
                            dir_recip_xyz_near_ab,
                        );
                        let dir_recip_z = shuffle_m256::<0b10_10_10_10>(
                            dir_recip_xyz_near_ab,
                            dir_recip_xyz_near_ab,
                        );

                        let ray_near_far = shuffle_m256::<0b11_11_11_11>(
                            dir_recip_xyz_near_ab,
                            origin_xyz_far_ab_neg,
                        );

                        let origin_dir_recip_x = origin_x * dir_recip_x;
                        let origin_dir_recip_y = origin_y * dir_recip_y;
                        let origin_dir_recip_z = origin_z * dir_recip_z;

                        let bb_near_far_x =
                            blend_varying_m256(bb_min_max_x, bb_max_min_x, dir_recip_x);
                        let bb_near_far_y =
                            blend_varying_m256(bb_min_max_y, bb_max_min_y, dir_recip_y);
                        let bb_near_far_z =
                            blend_varying_m256(bb_min_max_z, bb_max_min_z, dir_recip_z);

                        let tnear_far_x =
                            fused_mul_sub_m256(bb_near_far_x, dir_recip_x, origin_dir_recip_x);
                        let tnear_far_y =
                            fused_mul_sub_m256(bb_near_far_y, dir_recip_y, origin_dir_recip_y);
                        let tnear_far_z =
                            fused_mul_sub_m256(bb_near_far_z, dir_recip_z, origin_dir_recip_z);

                        // near 0, near 1, -far 0, -far 1
                        let tnear_far = max_m256(
                            max_m256(max_m256(tnear_far_x, tnear_far_y), tnear_far_z),
                            ray_near_far,
                        );

                        // near 0, near 1, near 0, near 0
                        // far 0, far 1, near 1, near 1
                        let mask =
                            move_mask_m256(cmp_op_mask_m256::<{ cmp_op!(LessEqualOrdered) }>(
                                shuffle_m256::<0b00_00_01_00>(tnear_far, tnear_far),
                                shuffle_m256::<0b01_01_11_10>(tnear_far ^ neg_mask, tnear_far),
                            ));

                        let left_hit_a = mask & 0b1;
                        let right_hit_a = (mask >> 1) & 0b1;

                        ray_lists[0][ray_list_sizes[0]] = ray_idx_a as RayIdx;
                        ray_lists[1][ray_list_sizes[1]] = ray_idx_a as RayIdx;
                        ray_list_sizes[0] += left_hit_a as usize;
                        ray_list_sizes[1] += right_hit_a as usize;

                        if ray_idx_a != ray_idx_b {
                            let left_hit_b = (mask >> 4) & 0b1;
                            let right_hit_b = (mask >> 5) & 0b1;

                            ray_lists[0][ray_list_sizes[0]] = ray_idx_b as RayIdx;
                            ray_lists[1][ray_list_sizes[1]] = ray_idx_b as RayIdx;
                            ray_list_sizes[0] += left_hit_b as usize;
                            ray_list_sizes[1] += right_hit_b as usize;
                        }

                        active_ray_idx += 2;
                    }

                    if ray_list_sizes[1] - ray_list_sizes_orig[1] > 0 {
                        stack.push((&children[1], 1, ray_list_sizes_orig[1]));
                    }

                    if ray_list_sizes[0] - ray_list_sizes_orig[0] > 0 {
                        stack.push((&children[0], 0, ray_list_sizes_orig[0]));
                    }
                }
                BvhNode::Leaf { triangles_range } => {
                    stats.leaf_visit(
                        (last_active_ray_idx - active_ray_idx) as u64,
                        triangles_range.len() as u64,
                    );

                    self.occluded_objs(
                        triangles_range.clone(),
                        &ray_lists[list_idx][active_ray_idx..last_active_ray_idx],
                        rays,
                        occluded,
                    );
                }
            }
        }
    }

    fn intersect_objs(
        &self,
        triangles_range: Range<usize>,
        ray_indices: &[RayIdx],
        rays: &mut [StreamRay],
        hit_objects: &mut [Option<RayIdx>],
    ) {
        for ray_chunk_indices in ray_indices.chunks(4) {
            let ray_indices_padded = [
                ray_chunk_indices[0] as usize,
                *ray_chunk_indices.get(1).unwrap_or(&ray_chunk_indices[0]) as usize,
                *ray_chunk_indices.get(2).unwrap_or(&ray_chunk_indices[0]) as usize,
                *ray_chunk_indices.get(3).unwrap_or(&ray_chunk_indices[0]) as usize,
            ];
            let ray_origins = Vec3x4f::from([
                rays[ray_indices_padded[0]].origin_far.xyz(),
                rays[ray_indices_padded[1]].origin_far.xyz(),
                rays[ray_indices_padded[2]].origin_far.xyz(),
                rays[ray_indices_padded[3]].origin_far.xyz(),
            ]);
            let ray_directions = Vec3x4f::from([
                rays[ray_indices_padded[0]].direction.xyz(),
                rays[ray_indices_padded[1]].direction.xyz(),
                rays[ray_indices_padded[2]].direction.xyz(),
                rays[ray_indices_padded[3]].direction.xyz(),
            ]);

            for tri_idx in triangles_range.clone() {
                let tri = &self.triangles[tri_idx];
                let hit = tri.intersect_simd(&ray_origins, &ray_directions);

                let hit: [f32; 4] = hit.into();
                for (&ray_idx, hit) in ray_chunk_indices.iter().zip(hit) {
                    let ray = &mut rays[ray_idx as usize];
                    if (ray.direction_recip_near.w..ray.origin_far.w).contains(&hit) {
                        ray.origin_far.w = hit;
                        hit_objects[ray_idx as usize] =
                            Some(self.object_indices[tri_idx] as RayIdx);
                    }
                }
            }
        }
    }

    fn occluded_objs(
        &self,
        triangles_range: Range<usize>,
        ray_indices: &[RayIdx],
        rays: &mut [StreamRay],
        occluded: &mut [bool],
    ) {
        for ray_chunk_indices in ray_indices.chunks(4) {
            let ray_indices_padded = [
                ray_chunk_indices[0] as usize,
                *ray_chunk_indices.get(1).unwrap_or(&ray_chunk_indices[0]) as usize,
                *ray_chunk_indices.get(2).unwrap_or(&ray_chunk_indices[0]) as usize,
                *ray_chunk_indices.get(3).unwrap_or(&ray_chunk_indices[0]) as usize,
            ];
            let ray_origins = Vec3x4f::from([
                rays[ray_indices_padded[0]].origin_far.xyz(),
                rays[ray_indices_padded[1]].origin_far.xyz(),
                rays[ray_indices_padded[2]].origin_far.xyz(),
                rays[ray_indices_padded[3]].origin_far.xyz(),
            ]);
            let ray_directions = Vec3x4f::from([
                rays[ray_indices_padded[0]].direction.xyz(),
                rays[ray_indices_padded[1]].direction.xyz(),
                rays[ray_indices_padded[2]].direction.xyz(),
                rays[ray_indices_padded[3]].direction.xyz(),
            ]);

            for tri_idx in triangles_range.clone() {
                let tri = &self.triangles[tri_idx];
                let hit = tri.intersect_simd(&ray_origins, &ray_directions);

                let hit: [f32; 4] = hit.into();
                for (&ray_idx, hit) in ray_chunk_indices.iter().zip(hit) {
                    let ray = &mut rays[ray_idx as usize];
                    if (ray.direction_recip_near.w..ray.origin_far.w).contains(&hit) {
                        ray.origin_far.w = hit;
                        occluded[ray_idx as usize] = true;
                    }
                }
            }
        }
    }
}
