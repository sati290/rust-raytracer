use std::ops::Add;
use std::ops::Range;

use crate::aabb::Aabb;
use crate::Vec3;
use crate::Vec3x4;
use crate::{Ray, TraceResult, Triangle};
use arrayvec::ArrayVec;
use wide::CmpLt;

#[derive(Clone, Copy, Debug)]
pub struct TraceStats {
    inner_visit: u32,
    leaf_visit: u32,
    obj_intersect: u32,
    obj_intersect_skipped: u32,
}

impl TraceStats {
    pub fn new() -> Self {
        TraceStats {
            inner_visit: 0,
            leaf_visit: 0,
            obj_intersect: 0,
            obj_intersect_skipped: 0,
        }
    }
}

impl Add for TraceStats {
    type Output = TraceStats;

    fn add(self, rhs: Self) -> Self::Output {
        TraceStats {
            inner_visit: self.inner_visit + rhs.inner_visit,
            leaf_visit: self.leaf_visit + rhs.leaf_visit,
            obj_intersect: self.obj_intersect + rhs.obj_intersect,
            obj_intersect_skipped: self.obj_intersect_skipped + rhs.obj_intersect_skipped,
        }
    }
}

enum BvhNode {
    Inner {
        child_bbox: [Aabb; 2],
        children: [Box<BvhNode>; 2],
    },
    Leaf {
        objects: Range<usize>,
    },
}

pub struct Bvh<'a> {
    objects: &'a [Triangle],
    object_indices: Vec<usize>,
    root_node: BvhNode,
}

impl Bvh<'_> {
    pub fn build(objects: &[Triangle]) -> Bvh {
        let mut object_indices: Vec<_> = (0..objects.len()).collect();
        let object_bounds: Vec<_> = objects.iter().map(|o| (o.centroid(), o.aabb())).collect();
        let root_node = Bvh::build_recursive(&mut object_indices, 0, &object_bounds);

        Bvh {
            objects,
            object_indices,
            root_node,
        }
    }

    fn build_recursive(
        indices: &mut [usize],
        indices_start_idx: usize,
        object_bounds: &[(Vec3, Aabb)],
    ) -> BvhNode {
        match indices.len() {
            1..=8 => BvhNode::Leaf {
                objects: indices_start_idx..indices_start_idx + indices.len(),
            },
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

                let child_left =
                    Bvh::build_recursive(indices_left, indices_start_idx, object_bounds);
                let child_right = Bvh::build_recursive(
                    indices_right,
                    indices_start_idx + split_idx,
                    object_bounds,
                );

                BvhNode::Inner {
                    child_bbox: [bounds_l, bounds_r],
                    children: [Box::new(child_left), Box::new(child_right)],
                }
            }
        }
    }

    pub fn trace_shadow(&self, ray_origin: &Vec3x4, ray_direction: &Vec3x4, ray_mask: i32) -> i32 {
        let ray_direction_recip = Vec3x4::splat(Vec3::broadcast(1.)) / *ray_direction;
        let mut result = !ray_mask;
        let mut stack = ArrayVec::<_, 32>::new();

        stack.push(&self.root_node);
        while let Some(node) = stack.pop() {
            match node {
                BvhNode::Inner {
                    child_bbox,
                    children,
                } => {
                    if child_bbox[0]
                        .intersect_simd(ray_origin, &ray_direction_recip)
                        .move_mask()
                        & !result
                        != 0
                    {
                        stack.push(&children[0]);
                    };
                    if child_bbox[1]
                        .intersect_simd(ray_origin, &ray_direction_recip)
                        .move_mask()
                        & !result
                        != 0
                    {
                        stack.push(&children[1]);
                    };
                }
                BvhNode::Leaf { objects } => {
                    for obj in objects
                        .clone()
                        .map(|i| &self.objects[self.object_indices[i]])
                    {
                        let hit = obj.intersect_simd::<false>(ray_origin, ray_direction);
                        result |= hit.cmp_lt(f32::INFINITY).move_mask();
                        if result == 0b1111 {
                            return result;
                        }
                    }
                }
            }
        }

        result
    }

    pub fn trace_stream<'a>(
        &'a self,
        rays: &[Ray],
        results: &mut [TraceResult<'a>],
        stats: &mut TraceStats,
    ) {
        use safe_arch::*;

        let mut stack = ArrayVec::<_, 32>::new();
        let mut ray_lists = [
            vec![0; rays.len() * 32],
            vec![0; rays.len() * 32],
            vec![0; rays.len() * 32],
        ];
        let mut ray_list_sizes = [0; 3];

        for i in 0..rays.len() {
            ray_lists[0][i] = i as u16;
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
                    stats.inner_visit += 1;

                    let bb_min_x = m128::from([
                        child_bbox[0].min.x,
                        child_bbox[1].min.x,
                        child_bbox[0].min.x,
                        child_bbox[1].min.x,
                    ]);
                    let bb_max_x = m128::from([
                        child_bbox[0].max.x,
                        child_bbox[1].max.x,
                        child_bbox[0].max.x,
                        child_bbox[1].max.x,
                    ]);
                    let bb_min_y = m128::from([
                        child_bbox[0].min.y,
                        child_bbox[1].min.y,
                        child_bbox[0].min.y,
                        child_bbox[1].min.y,
                    ]);
                    let bb_max_y = m128::from([
                        child_bbox[0].max.y,
                        child_bbox[1].max.y,
                        child_bbox[0].max.y,
                        child_bbox[1].max.y,
                    ]);
                    let bb_min_z = m128::from([
                        child_bbox[0].min.z,
                        child_bbox[1].min.z,
                        child_bbox[0].min.z,
                        child_bbox[1].min.z,
                    ]);
                    let bb_max_z = m128::from([
                        child_bbox[0].max.z,
                        child_bbox[1].max.z,
                        child_bbox[0].max.z,
                        child_bbox[1].max.z,
                    ]);

                    let ray_list_sizes_orig = ray_list_sizes;
                    while active_ray_idx < last_active_ray_idx {
                        let ray_idx_a = ray_lists[list_idx][active_ray_idx] as usize;
                        let ray_idx_b = if active_ray_idx + 1 < last_active_ray_idx {
                            ray_lists[list_idx][active_ray_idx + 1] as usize
                        } else {
                            ray_idx_a
                        };

                        let origin_xyzw_a = m128::from(*rays[ray_idx_a].origin.as_array());
                        let origin_xyzw_b = m128::from(*rays[ray_idx_b].origin.as_array());

                        let dir_recip_xyzw_a =
                            m128::from(*rays[ray_idx_a].direction_recip.as_array());
                        let dir_recip_xyzw_b =
                            m128::from(*rays[ray_idx_b].direction_recip.as_array());

                        let origin_x =
                            shuffle_abi_f32_all_m128::<0b00_00_00_00>(origin_xyzw_a, origin_xyzw_b);
                        let origin_y =
                            shuffle_abi_f32_all_m128::<0b01_01_01_01>(origin_xyzw_a, origin_xyzw_b);
                        let origin_z =
                            shuffle_abi_f32_all_m128::<0b10_10_10_10>(origin_xyzw_a, origin_xyzw_b);

                        let dir_recip_x = shuffle_abi_f32_all_m128::<0b00_00_00_00>(
                            dir_recip_xyzw_a,
                            dir_recip_xyzw_b,
                        );
                        let dir_recip_y = shuffle_abi_f32_all_m128::<0b01_01_01_01>(
                            dir_recip_xyzw_a,
                            dir_recip_xyzw_b,
                        );
                        let dir_recip_z = shuffle_abi_f32_all_m128::<0b10_10_10_10>(
                            dir_recip_xyzw_a,
                            dir_recip_xyzw_b,
                        );

                        let origin_dir_recip_x = origin_x * dir_recip_x;
                        let origin_dir_recip_y = origin_y * dir_recip_y;
                        let origin_dir_recip_z = origin_z * dir_recip_z;

                        let tx1 = fused_mul_sub_m128(bb_min_x, dir_recip_x, origin_dir_recip_x);
                        let tx2 = fused_mul_sub_m128(bb_max_x, dir_recip_x, origin_dir_recip_x);
                        let ty1 = fused_mul_sub_m128(bb_min_y, dir_recip_y, origin_dir_recip_y);
                        let ty2 = fused_mul_sub_m128(bb_max_y, dir_recip_y, origin_dir_recip_y);
                        let tz1 = fused_mul_sub_m128(bb_min_z, dir_recip_z, origin_dir_recip_z);
                        let tz2 = fused_mul_sub_m128(bb_max_z, dir_recip_z, origin_dir_recip_z);

                        let tnear = max_m128(
                            max_m128(min_m128(tx1, tx2), min_m128(ty1, ty2)),
                            min_m128(tz1, tz2),
                        );
                        let tfar = min_m128(
                            min_m128(max_m128(tx1, tx2), max_m128(ty1, ty2)),
                            max_m128(tz1, tz2),
                        );

                        let mask =
                            move_mask_m128(cmp_ge_mask_m128(tfar, max_m128(tnear, zeroed_m128())));

                        ray_lists[0][ray_list_sizes[0]] = ray_idx_a as u16;
                        ray_list_sizes[0] += (mask & 0b1) as usize;
                        ray_lists[1][ray_list_sizes[1]] = ray_idx_a as u16;
                        ray_list_sizes[1] += ((mask >> 1) & 0b1) as usize;
                        ray_lists[0][ray_list_sizes[0]] = ray_idx_b as u16;
                        ray_list_sizes[0] += ((mask >> 2) & 0b1) as usize;
                        ray_lists[1][ray_list_sizes[1]] = ray_idx_b as u16;
                        ray_list_sizes[1] += ((mask >> 3) & 0b1) as usize;

                        active_ray_idx += 2;
                    }

                    if ray_list_sizes[0] - ray_list_sizes_orig[0] > 0 {
                        stack.push((&children[0], 0, ray_list_sizes_orig[0]));
                    }

                    if ray_list_sizes[1] - ray_list_sizes_orig[1] > 0 {
                        stack.push((&children[1], 1, ray_list_sizes_orig[1]));
                    }
                }
                BvhNode::Leaf { objects } => {
                    stats.leaf_visit += 1;
                    stats.obj_intersect +=
                        ((last_active_ray_idx - active_ray_idx) * objects.len()) as u32;
                    stats.obj_intersect_skipped +=
                        stats.obj_intersect - rays.len() as u32 * objects.len() as u32;

                    for ray_indices in
                        ray_lists[list_idx][active_ray_idx..last_active_ray_idx].chunks(4)
                    {
                        let ray_indices_padded = [
                            ray_indices[0] as usize,
                            *ray_indices.get(1).unwrap_or(&ray_indices[0]) as usize,
                            *ray_indices.get(2).unwrap_or(&ray_indices[0]) as usize,
                            *ray_indices.get(3).unwrap_or(&ray_indices[0]) as usize,
                        ];
                        let ray_origins = Vec3x4::from([
                            rays[ray_indices_padded[0]].origin.xyz(),
                            rays[ray_indices_padded[1]].origin.xyz(),
                            rays[ray_indices_padded[2]].origin.xyz(),
                            rays[ray_indices_padded[3]].origin.xyz(),
                        ]);
                        let ray_directions = Vec3x4::from([
                            rays[ray_indices_padded[0]].direction.xyz(),
                            rays[ray_indices_padded[1]].direction.xyz(),
                            rays[ray_indices_padded[2]].direction.xyz(),
                            rays[ray_indices_padded[3]].direction.xyz(),
                        ]);

                        for obj in objects
                            .clone()
                            .map(|i| &self.objects[self.object_indices[i]])
                        {
                            let hit = obj.intersect_simd::<false>(&ray_origins, &ray_directions);

                            let hit: [f32; 4] = hit.into();
                            for (&i, hit) in ray_indices.iter().zip(hit) {
                                results[i as usize].add_hit(hit, obj);
                            }
                        }
                    }
                }
            }
        }
    }
}
