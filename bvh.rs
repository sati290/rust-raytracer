use std::ops::Add;
use std::ops::Range;

use crate::aabb::Aabb;
use crate::Vec3;
use crate::Vec3x4;
use crate::{Ray, TraceResult, Triangle};
use arrayvec::ArrayVec;
use wide::CmpLt;

#[inline]
#[allow(non_snake_case)]
pub const fn _MM_SHUFFLE_(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

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

                    let ray_list_sizes_orig = ray_list_sizes;
                    while active_ray_idx < last_active_ray_idx {
                        let ray_indices = [
                            ray_lists[list_idx][active_ray_idx] as usize,
                            if active_ray_idx + 1 < last_active_ray_idx {
                                ray_lists[list_idx][active_ray_idx + 1]
                            } else {
                                ray_lists[list_idx][active_ray_idx]
                            } as usize,
                            if active_ray_idx + 2 < last_active_ray_idx {
                                ray_lists[list_idx][active_ray_idx + 2]
                            } else {
                                ray_lists[list_idx][active_ray_idx]
                            } as usize,
                            if active_ray_idx + 3 < last_active_ray_idx {
                                ray_lists[list_idx][active_ray_idx + 3]
                            } else {
                                ray_lists[list_idx][active_ray_idx]
                            } as usize,
                        ];

                        let mut origin_x = m128::from(*rays[ray_indices[0]].origin.as_array());
                        let mut origin_y = m128::from(*rays[ray_indices[1]].origin.as_array());
                        let mut origin_z = m128::from(*rays[ray_indices[2]].origin.as_array());
                        let mut origin_w = m128::from(*rays[ray_indices[3]].origin.as_array());

                        transpose_four_m128(
                            &mut origin_x,
                            &mut origin_y,
                            &mut origin_z,
                            &mut origin_w,
                        );

                        let mut dir_recip_x =
                            m128::from(*rays[ray_indices[0]].direction_recip.as_array());
                        let mut dir_recip_y =
                            m128::from(*rays[ray_indices[1]].direction_recip.as_array());
                        let mut dir_recip_z =
                            m128::from(*rays[ray_indices[2]].direction_recip.as_array());
                        let mut dir_recip_w =
                            m128::from(*rays[ray_indices[3]].direction_recip.as_array());

                        transpose_four_m128(
                            &mut dir_recip_x,
                            &mut dir_recip_y,
                            &mut dir_recip_z,
                            &mut dir_recip_w,
                        );

                        let origin_dir_recip_x = origin_x * dir_recip_x;
                        let origin_dir_recip_y = origin_y * dir_recip_y;
                        let origin_dir_recip_z = origin_z * dir_recip_z;

                        let intersect_bb = |aabb: &Aabb| {
                            let bb_min_x = load_f32_splat_m128(&aabb.min.x);
                            let bb_max_x = load_f32_splat_m128(&aabb.max.x);
                            let bb_min_y = load_f32_splat_m128(&aabb.min.y);
                            let bb_max_y = load_f32_splat_m128(&aabb.max.y);
                            let bb_min_z = load_f32_splat_m128(&aabb.min.z);
                            let bb_max_z = load_f32_splat_m128(&aabb.max.z);
                            let tx1 = fused_mul_sub_m128(bb_min_x, dir_recip_x, origin_dir_recip_x);
                            let tx2 = fused_mul_sub_m128(bb_max_x, dir_recip_x, origin_dir_recip_x);

                            let tmin = min_m128(tx1, tx2);
                            let tmax = max_m128(tx1, tx2);

                            let ty1 = fused_mul_sub_m128(bb_min_y, dir_recip_y, origin_dir_recip_y);
                            let ty2 = fused_mul_sub_m128(bb_max_y, dir_recip_y, origin_dir_recip_y);

                            let tmin = max_m128(tmin, min_m128(ty1, ty2));
                            let tmax = min_m128(tmax, max_m128(ty1, ty2));

                            let tz1 = fused_mul_sub_m128(bb_min_z, dir_recip_z, origin_dir_recip_z);
                            let tz2 = fused_mul_sub_m128(bb_max_z, dir_recip_z, origin_dir_recip_z);

                            let tmin = max_m128(tmin, min_m128(tz1, tz2));
                            let tmax = min_m128(tmax, max_m128(tz1, tz2));

                            move_mask_m128(cmp_ge_mask_m128(tmax, max_m128(tmin, zeroed_m128())))
                        };

                        let hit_left = intersect_bb(&child_bbox[0]);
                        let hit_right = intersect_bb(&child_bbox[1]);

                        // let hit_left = child_bbox[0]
                        //     .intersect_simd(&origins, &directions_recip)
                        //     .move_mask();
                        // let hit_right = child_bbox[1]
                        //     .intersect_simd(&origins, &directions_recip)
                        //     .move_mask();
                        for (i, ray_idx) in ray_indices
                            .iter()
                            .enumerate()
                            .take(last_active_ray_idx - active_ray_idx)
                        {
                            if hit_left & 1 << i != 0 {
                                ray_lists[0][ray_list_sizes[0]] = *ray_idx as u16;
                                ray_list_sizes[0] += 1;
                            }
                            if hit_right & 1 << i != 0 {
                                ray_lists[1][ray_list_sizes[1]] = *ray_idx as u16;
                                ray_list_sizes[1] += 1;
                            }
                        }

                        active_ray_idx += 4;
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
