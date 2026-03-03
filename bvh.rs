use arrayvec::ArrayVec;
use std::ops::{Add, AddAssign, Range};
use ultraviolet::{Vec3, Vec3x4};

use crate::Ray;
use crate::aabb::Aabb;
use crate::trace_stats::TraceStats;
use crate::triangle::Triangle;
use crate::triangle_opt::TriangleOpt;

type RayIdx = u32;

#[derive(Debug)]
struct BvhStats {
    num_leaves: u32,
    max_depth: u32,
}

impl BvhStats {
    fn new() -> Self {
        BvhStats {
            num_leaves: 0,
            max_depth: 0,
        }
    }

    fn add_leaf(&mut self, depth: u32) {
        self.num_leaves += 1;
        self.max_depth = self.max_depth.max(depth);
    }
}

impl Add for BvhStats {
    type Output = BvhStats;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl AddAssign for BvhStats {
    fn add_assign(&mut self, rhs: Self) {
        self.num_leaves += rhs.num_leaves;
        self.max_depth = self.max_depth.max(rhs.max_depth);
    }
}

enum BvhNode {
    Inner {
        // minl, minr, maxl, maxr
        child_bbox: Vec3x4,
        children: [Box<BvhNode>; 2],
    },
    Leaf {
        triangles_range: Range<usize>,
    },
}

pub struct Bvh {
    triangles: Vec<TriangleOpt>,
    object_indices: Vec<usize>,
    root_node: BvhNode,
}

impl Bvh {
    pub fn build(objects: &[Triangle]) -> Bvh {
        let mut object_indices: Vec<_> = (0..objects.len()).collect();
        let object_bounds: Vec<_> = objects.iter().map(|o| (o.centroid(), o.aabb())).collect();

        let mut stats = BvhStats::new();
        let root_node = Bvh::build_recursive(&mut object_indices, 0, &object_bounds, 0, &mut stats);
        println!("{:?}", stats);

        let triangles = object_indices
            .iter()
            .map(|&idx| TriangleOpt::from(&objects[idx]))
            .collect();

        Bvh {
            triangles,
            object_indices,
            root_node,
        }
    }

    fn build_recursive(
        indices: &mut [usize],
        indices_start_idx: usize,
        object_bounds: &[(Vec3, Aabb)],
        depth: u32,
        stats: &mut BvhStats,
    ) -> BvhNode {
        match indices.len() {
            1..=8 => {
                stats.add_leaf(depth);

                BvhNode::Leaf {
                    triangles_range: indices_start_idx..indices_start_idx + indices.len(),
                }
            }
            _ => {
                let mut bounds = Aabb::empty();
                let mut centroid_bounds = Aabb::empty();
                for idx in &*indices {
                    let (centroid, aabb) = &object_bounds[*idx];
                    bounds.join_mut(*aabb);
                    centroid_bounds.grow_mut(*centroid);
                }

                let size = centroid_bounds.size();
                let largest_axis = if size.x > size.y && size.x > size.z {
                    0
                } else if size.y > size.z {
                    1
                } else {
                    2
                };

                const BUCKET_COUNT: usize = 8;
                let mut buckets = [(Aabb::empty(), 0u32); BUCKET_COUNT];

                let k0 = centroid_bounds.min[largest_axis];
                let k1 = BUCKET_COUNT as f32 * (1. - 0.01)
                    / (centroid_bounds.max[largest_axis] - centroid_bounds.min[largest_axis]);
                let get_bucket_idx =
                    |centroid: &Vec3| (k1 * (centroid[largest_axis] - k0)) as usize;
                for idx in &*indices {
                    let (centroid, aabb) = &object_bounds[*idx];
                    let bucket_idx = get_bucket_idx(centroid);
                    let bucket = &mut buckets[bucket_idx];

                    bucket.0.join_mut(*aabb);
                    bucket.1 += 1;
                }

                let mut min_bucket = 0;
                let mut min_cost = f32::INFINITY;
                let mut bounds_l = Aabb::empty();
                let mut bounds_r = Aabb::empty();
                for i in 0..(BUCKET_COUNT - 1) {
                    let (buckets_l, buckets_r) = buckets.split_at(i + 1);
                    let child_l = buckets_l.iter().fold((Aabb::empty(), 0u32), |acc, x| {
                        (acc.0.join(x.0), acc.1 + x.1)
                    });
                    let child_r = buckets_r.iter().fold((Aabb::empty(), 0u32), |acc, x| {
                        (acc.0.join(x.0), acc.1 + x.1)
                    });

                    let cost = (child_l.0.surface_area() * child_l.1 as f32
                        + child_r.0.surface_area() * child_r.1 as f32)
                        / bounds.surface_area();
                    if cost < min_cost {
                        min_bucket = i;
                        min_cost = cost;
                        bounds_l = child_l.0;
                        bounds_r = child_r.0;
                    }
                }

                let mut left = 0;
                let mut right = indices.len() - 1;
                loop {
                    while get_bucket_idx(&object_bounds[indices[left]].0) <= min_bucket {
                        left += 1;
                    }

                    while get_bucket_idx(&object_bounds[indices[right]].0) > min_bucket {
                        right -= 1;
                    }

                    if left >= right {
                        break;
                    }

                    indices.swap(left, right);
                }

                let split_idx = left;
                let (indices_left, indices_right) = indices.split_at_mut(split_idx);

                let mut stats_r = BvhStats::new();
                let (child_left, child_right) = rayon::join(
                    || {
                        Bvh::build_recursive(
                            indices_left,
                            indices_start_idx,
                            object_bounds,
                            depth + 1,
                            stats,
                        )
                    },
                    || {
                        Bvh::build_recursive(
                            indices_right,
                            indices_start_idx + split_idx,
                            object_bounds,
                            depth + 1,
                            &mut stats_r,
                        )
                    },
                );
                *stats += stats_r;

                BvhNode::Inner {
                    child_bbox: Vec3x4::from([
                        bounds_l.min,
                        bounds_r.min,
                        bounds_l.max,
                        bounds_r.max,
                    ]),
                    children: [Box::new(child_left), Box::new(child_right)],
                }
            }
        }
    }

    pub fn occluded1(&self, ray: &Ray, stats: &mut TraceStats) -> bool {
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

                    let tnear = max_m128(max_m128(tnear_far_x, tnear_far_y), tnear_far_z);
                    let tfar = min_m128(min_m128(tnear_far_x, tnear_far_y), tnear_far_z);

                    let tnear_far = shuffle_abi_f32_all_m128::<0b_11_10_01_00>(
                        max_m128(tnear, zeroed_m128()),
                        tfar,
                    )
                    .to_array();

                    let hit_l = tnear_far[0] < tnear_far[2];
                    let hit_r = tnear_far[1] < tnear_far[3];

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
                        if (ray.direction_recip_near.w..f32::INFINITY).contains(&hit) {
                            return true;
                        }
                    }

                    cur_node = stack.pop();
                }
            }
        }

        false
    }

    pub fn intersect_stream(
        &self,
        rays: &mut [Ray],
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
                    let bb_min_max_x = load_m128_splat_m256(&m128::from(child_bbox.x.to_array()));
                    let bb_min_max_y = load_m128_splat_m256(&m128::from(child_bbox.y.to_array()));
                    let bb_min_max_z = load_m128_splat_m256(&m128::from(child_bbox.z.to_array()));

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
                            m128::from(*rays[ray_idx_b].origin_far.as_array()),
                            m128::from(*rays[ray_idx_a].origin_far.as_array()),
                        );

                        let origin_xyz_far_ab_neg = origin_xyz_far_ab ^ neg_mask;

                        let dir_recip_xyz_near_ab = set_m128_m256(
                            m128::from(*rays[ray_idx_b].direction_recip_near.as_array()),
                            m128::from(*rays[ray_idx_a].direction_recip_near.as_array()),
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

    pub fn occluded_stream(&self, rays: &mut [Ray], occluded: &mut [bool], stats: &mut TraceStats) {
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
                    let bb_min_max_x = load_m128_splat_m256(&m128::from(child_bbox.x.to_array()));
                    let bb_min_max_y = load_m128_splat_m256(&m128::from(child_bbox.y.to_array()));
                    let bb_min_max_z = load_m128_splat_m256(&m128::from(child_bbox.z.to_array()));

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
                            m128::from(*rays[ray_idx_b].origin_far.as_array()),
                            m128::from(*rays[ray_idx_a].origin_far.as_array()),
                        );

                        let origin_xyz_far_ab_neg = origin_xyz_far_ab ^ neg_mask;

                        let dir_recip_xyz_near_ab = set_m128_m256(
                            m128::from(*rays[ray_idx_b].direction_recip_near.as_array()),
                            m128::from(*rays[ray_idx_a].direction_recip_near.as_array()),
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
        rays: &mut [Ray],
        hit_objects: &mut [Option<RayIdx>],
    ) {
        for ray_chunk_indices in ray_indices.chunks(4) {
            let ray_indices_padded = [
                ray_chunk_indices[0] as usize,
                *ray_chunk_indices.get(1).unwrap_or(&ray_chunk_indices[0]) as usize,
                *ray_chunk_indices.get(2).unwrap_or(&ray_chunk_indices[0]) as usize,
                *ray_chunk_indices.get(3).unwrap_or(&ray_chunk_indices[0]) as usize,
            ];
            let ray_origins = Vec3x4::from([
                rays[ray_indices_padded[0]].origin_far.xyz(),
                rays[ray_indices_padded[1]].origin_far.xyz(),
                rays[ray_indices_padded[2]].origin_far.xyz(),
                rays[ray_indices_padded[3]].origin_far.xyz(),
            ]);
            let ray_directions = Vec3x4::from([
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
        rays: &mut [Ray],
        occluded: &mut [bool],
    ) {
        for ray_chunk_indices in ray_indices.chunks(4) {
            let ray_indices_padded = [
                ray_chunk_indices[0] as usize,
                *ray_chunk_indices.get(1).unwrap_or(&ray_chunk_indices[0]) as usize,
                *ray_chunk_indices.get(2).unwrap_or(&ray_chunk_indices[0]) as usize,
                *ray_chunk_indices.get(3).unwrap_or(&ray_chunk_indices[0]) as usize,
            ];
            let ray_origins = Vec3x4::from([
                rays[ray_indices_padded[0]].origin_far.xyz(),
                rays[ray_indices_padded[1]].origin_far.xyz(),
                rays[ray_indices_padded[2]].origin_far.xyz(),
                rays[ray_indices_padded[3]].origin_far.xyz(),
            ]);
            let ray_directions = Vec3x4::from([
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
