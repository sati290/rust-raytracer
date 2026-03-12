use arrayvec::ArrayVec;

use crate::{bvh::*, ray::StreamRay, trace_stats::TraceStats};

impl Bvh {
    #[must_use]
    pub fn occluded1(&self, ray: &StreamRay, stats: &mut TraceStats) -> bool {
        use safe_arch::*;

        stats.trace_start(1);

        let mut stack = ArrayVec::<&BvhNode, 32>::new();

        let origin_xyz_far = m128::from(*ray.origin_far.as_array());
        let origin_x = shuffle_abi_f32_all_m128::<0b00_00_00_00>(origin_xyz_far, origin_xyz_far);
        let origin_y = shuffle_abi_f32_all_m128::<0b01_01_01_01>(origin_xyz_far, origin_xyz_far);
        let origin_z = shuffle_abi_f32_all_m128::<0b10_10_10_10>(origin_xyz_far, origin_xyz_far);
        let dir_recip_xyz_near = m128::from(*ray.direction_recip_near.as_array());
        let dir_recip_x =
            shuffle_abi_f32_all_m128::<0b00_00_00_00>(dir_recip_xyz_near, dir_recip_xyz_near);
        let dir_recip_y =
            shuffle_abi_f32_all_m128::<0b01_01_01_01>(dir_recip_xyz_near, dir_recip_xyz_near);
        let dir_recip_z =
            shuffle_abi_f32_all_m128::<0b10_10_10_10>(dir_recip_xyz_near, dir_recip_xyz_near);
        let ray_near_far =
            shuffle_abi_f32_all_m128::<0b11_11_11_11>(dir_recip_xyz_near, origin_xyz_far);

        let origin_dir_recip_x = origin_x * dir_recip_x;
        let origin_dir_recip_y = origin_y * dir_recip_y;
        let origin_dir_recip_z = origin_z * dir_recip_z;

        let mut cur_node = Some(&self.root_node);
        while let Some(node) = cur_node {
            match node {
                BvhNode::Inner {
                    child_bbox,
                    children,
                } => {
                    stats.inner_visit(1);

                    let bb_min_max_x = m128::from(child_bbox.x.to_array());
                    let bb_min_max_y = m128::from(child_bbox.y.to_array());
                    let bb_min_max_z = m128::from(child_bbox.z.to_array());

                    let bb_max_min_x =
                        shuffle_abi_f32_all_m128::<0b01_00_11_10>(bb_min_max_x, bb_min_max_x);
                    let bb_max_min_y =
                        shuffle_abi_f32_all_m128::<0b01_00_11_10>(bb_min_max_y, bb_min_max_y);
                    let bb_max_min_z =
                        shuffle_abi_f32_all_m128::<0b01_00_11_10>(bb_min_max_z, bb_min_max_z);

                    let bb_near_far_x = blend_varying_m128(bb_min_max_x, bb_max_min_x, dir_recip_x);
                    let bb_near_far_y = blend_varying_m128(bb_min_max_y, bb_max_min_y, dir_recip_y);
                    let bb_near_far_z = blend_varying_m128(bb_min_max_z, bb_max_min_z, dir_recip_z);

                    let tnear_far_x =
                        fused_mul_sub_m128(bb_near_far_x, dir_recip_x, origin_dir_recip_x);
                    let tnear_far_y =
                        fused_mul_sub_m128(bb_near_far_y, dir_recip_y, origin_dir_recip_y);
                    let tnear_far_z =
                        fused_mul_sub_m128(bb_near_far_z, dir_recip_z, origin_dir_recip_z);

                    let tnear = max_m128(
                        max_m128(tnear_far_x, tnear_far_y),
                        max_m128(tnear_far_z, ray_near_far),
                    );
                    let tfar = min_m128(
                        min_m128(tnear_far_x, tnear_far_y),
                        min_m128(tnear_far_z, ray_near_far),
                    );

                    let tnear_far =
                        shuffle_abi_f32_all_m128::<0b_11_10_01_00>(tnear, tfar).to_array();

                    let hit_l = tnear_far[0] <= tnear_far[2];
                    let hit_r = tnear_far[1] <= tnear_far[3];

                    if hit_l {
                        cur_node = Some(&children[0]);

                        if hit_r {
                            stack.push(&children[1]);
                        }
                    } else if hit_r {
                        cur_node = Some(&children[1]);
                    } else {
                        cur_node = stack.pop();
                    }
                }
                BvhNode::Leaf { triangles_range } => {
                    stats.leaf_visit(1, triangles_range.len() as u64);
                    for obj_idx in triangles_range.clone() {
                        let obj = &self.triangles[obj_idx];
                        let hit = obj.intersect(&ray.origin_far.xyz(), &ray.direction.xyz());
                        if (ray.direction_recip_near.w..ray.origin_far.w).contains(&hit) {
                            return true;
                        }
                    }

                    cur_node = stack.pop();
                }
            }
        }

        false
    }

    #[must_use]
    pub fn intersect1(&self, ray: &mut StreamRay, stats: &mut TraceStats) -> Option<u32> {
        use safe_arch::*;

        stats.trace_start(1);

        let mut hit_obj = None;
        let mut stack = ArrayVec::<&BvhNode, 32>::new();

        let origin_xyz_far = m128::from(*ray.origin_far.as_array());
        let origin_x = shuffle_abi_f32_all_m128::<0b00_00_00_00>(origin_xyz_far, origin_xyz_far);
        let origin_y = shuffle_abi_f32_all_m128::<0b01_01_01_01>(origin_xyz_far, origin_xyz_far);
        let origin_z = shuffle_abi_f32_all_m128::<0b10_10_10_10>(origin_xyz_far, origin_xyz_far);
        let dir_recip_xyz_near = m128::from(*ray.direction_recip_near.as_array());
        let dir_recip_x =
            shuffle_abi_f32_all_m128::<0b00_00_00_00>(dir_recip_xyz_near, dir_recip_xyz_near);
        let dir_recip_y =
            shuffle_abi_f32_all_m128::<0b01_01_01_01>(dir_recip_xyz_near, dir_recip_xyz_near);
        let dir_recip_z =
            shuffle_abi_f32_all_m128::<0b10_10_10_10>(dir_recip_xyz_near, dir_recip_xyz_near);
        let ray_near_far =
            shuffle_abi_f32_all_m128::<0b11_11_11_11>(dir_recip_xyz_near, origin_xyz_far);

        let origin_dir_recip_x = origin_x * dir_recip_x;
        let origin_dir_recip_y = origin_y * dir_recip_y;
        let origin_dir_recip_z = origin_z * dir_recip_z;

        let mut cur_node = Some(&self.root_node);
        while let Some(node) = cur_node {
            match node {
                BvhNode::Inner {
                    child_bbox,
                    children,
                } => {
                    stats.inner_visit(1);

                    let bb_min_max_x = m128::from(child_bbox.x.to_array());
                    let bb_min_max_y = m128::from(child_bbox.y.to_array());
                    let bb_min_max_z = m128::from(child_bbox.z.to_array());

                    let bb_max_min_x =
                        shuffle_abi_f32_all_m128::<0b01_00_11_10>(bb_min_max_x, bb_min_max_x);
                    let bb_max_min_y =
                        shuffle_abi_f32_all_m128::<0b01_00_11_10>(bb_min_max_y, bb_min_max_y);
                    let bb_max_min_z =
                        shuffle_abi_f32_all_m128::<0b01_00_11_10>(bb_min_max_z, bb_min_max_z);

                    let bb_near_far_x = blend_varying_m128(bb_min_max_x, bb_max_min_x, dir_recip_x);
                    let bb_near_far_y = blend_varying_m128(bb_min_max_y, bb_max_min_y, dir_recip_y);
                    let bb_near_far_z = blend_varying_m128(bb_min_max_z, bb_max_min_z, dir_recip_z);

                    let tnear_far_x =
                        fused_mul_sub_m128(bb_near_far_x, dir_recip_x, origin_dir_recip_x);
                    let tnear_far_y =
                        fused_mul_sub_m128(bb_near_far_y, dir_recip_y, origin_dir_recip_y);
                    let tnear_far_z =
                        fused_mul_sub_m128(bb_near_far_z, dir_recip_z, origin_dir_recip_z);

                    let tnear = max_m128(
                        max_m128(tnear_far_x, tnear_far_y),
                        max_m128(tnear_far_z, ray_near_far),
                    );
                    let tfar = min_m128(
                        min_m128(tnear_far_x, tnear_far_y),
                        min_m128(tnear_far_z, ray_near_far),
                    );

                    let tnear_far =
                        shuffle_abi_f32_all_m128::<0b_11_10_01_00>(tnear, tfar).to_array();

                    let hit_l = tnear_far[0] <= tnear_far[2];
                    let hit_r = tnear_far[1] <= tnear_far[3];

                    if hit_l {
                        cur_node = Some(&children[0]);

                        if hit_r {
                            stack.push(&children[1]);
                        }
                    } else if hit_r {
                        cur_node = Some(&children[1]);
                    } else {
                        cur_node = stack.pop();
                    }
                }
                BvhNode::Leaf { triangles_range } => {
                    stats.leaf_visit(1, triangles_range.len() as u64);
                    for obj_idx in triangles_range.clone() {
                        let obj = &self.triangles[obj_idx];
                        let hit = obj.intersect(&ray.origin_far.xyz(), &ray.direction.xyz());
                        if (ray.direction_recip_near.w..ray.origin_far.w).contains(&hit) {
                            ray.origin_far.w = hit;
                            hit_obj = Some(self.object_indices[obj_idx] as u32);
                        }
                    }

                    cur_node = stack.pop();
                }
            }
        }

        hit_obj
    }
}
