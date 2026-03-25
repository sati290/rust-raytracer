use std::marker::PhantomData;

use arrayvec::ArrayVec;
use nalgebra::{SimdBool, SimdRealField};

use crate::{
    bvh::{
        node_intersector::{BvhNodeIntersector, BvhNodeIntersectorType},
        simd_ray::SimdRay,
        *,
    },
    math::SimdType,
    ray::{Ray, RayHit},
    trace_stats::TraceStats,
    triangle_intersector::TriangleIntersector,
};

pub struct BvhIntersector<T> {
    _phantom: PhantomData<T>,
}

impl<T> BvhIntersector<T>
where
    T: SimdRealField<Element = f32> + Copy + BvhNodeIntersectorType + SimdType,
{
    #[must_use]
    pub fn occluded(bvh: &Bvh, ray: &Ray<T>, stats: &mut TraceStats) -> T::SimdBool {
        stats.trace_start(ray.valid.bitmask().count_ones() as u64);

        let mut active = ray.valid;
        let mut stack = ArrayVec::<_, 32>::new();
        let simd_ray = <<T as BvhNodeIntersectorType>::BvhNodeIntersector as BvhNodeIntersector<
            T,
        >>::SimdRay::from(ray);

        stack.push((&bvh.root_node, active));
        'pop: while let Some((node, node_active)) = stack.pop() {
            if (node_active & active).none() {
                continue;
            }

            let num_valid_rays = node_active.bitmask().count_ones();
            let mut cur_node = node;
            while let BvhNode::Inner {
                child_bbox,
                children,
            } = cur_node
            {
                stats.inner_visit(num_valid_rays as u64);

                let mut next_node = None;
                for (i, child) in children.iter().enumerate() {
                    let (hit, _) = T::BvhNodeIntersector::intersect(child_bbox, i, &simd_ray);

                    if hit.any() {
                        if next_node.is_some() {
                            stack.push((child, hit));
                        } else {
                            next_node = Some(child);
                        }
                    }
                }

                if let Some(node) = next_node {
                    cur_node = node;
                } else {
                    continue 'pop;
                }
            }

            if let BvhNode::Leaf { triangles_range } = cur_node {
                stats.leaf_visit(num_valid_rays as u64, triangles_range.len() as u64);
                for obj in bvh.triangles[triangles_range.clone()].iter() {
                    let hit = TriangleIntersector::<T>::intersect(obj, &ray.origin, &ray.direction);
                    let hit_mask = hit.simd_ge(ray.near) & hit.simd_lt(ray.far);
                    active = active & !hit_mask;
                    if active.none() {
                        return !active & ray.valid;
                    }
                }
            }
        }

        !active & ray.valid
    }

    pub fn intersect(bvh: &Bvh, ray_hit: &mut RayHit<T>, stats: &mut TraceStats) {
        stats.trace_start(ray_hit.ray.valid.bitmask().count_ones() as u64);

        let mut stack = ArrayVec::<_, 64>::new();
        let mut simd_ray = <<T as BvhNodeIntersectorType>::BvhNodeIntersector as BvhNodeIntersector<
            T,
        >>::SimdRay::from(&ray_hit.ray);

        stack.push((&bvh.root_node, ray_hit.ray.near));
        'pop: while let Some((node, node_near)) = stack.pop() {
            let node_ray_valid = node_near.simd_le(ray_hit.ray.far);
            if node_ray_valid.none() {
                continue;
            }

            let num_valid_rays = node_ray_valid.bitmask().count_ones();
            let mut cur_node = node;
            while let BvhNode::Inner {
                child_bbox,
                children,
            } = cur_node
            {
                stats.inner_visit(num_valid_rays as u64);

                let mut next_node = None;
                let mut next_dist = T::splat(f32::INFINITY);
                for (i, child_node) in children.iter().enumerate() {
                    let (hit, near) = T::BvhNodeIntersector::intersect(child_bbox, i, &simd_ray);

                    if hit.any() {
                        let dist = hit.if_else(|| near, || T::splat(f32::INFINITY));
                        if dist.simd_lt(next_dist).any() {
                            if let Some(next_node) = next_node {
                                stack.push((next_node, next_dist));
                            }
                            next_node = Some(child_node);
                            next_dist = dist;
                        } else {
                            debug_assert!(next_node.is_some());
                            stack.push((child_node, dist));
                        }
                    }
                }

                if let Some(node) = next_node {
                    cur_node = node;
                } else {
                    continue 'pop;
                }
            }

            if let BvhNode::Leaf { triangles_range } = cur_node {
                stats.leaf_visit(num_valid_rays as u64, triangles_range.len() as u64);
                for (obj, &obj_idx) in bvh.triangles[triangles_range.clone()]
                    .iter()
                    .zip(bvh.object_indices[triangles_range.clone()].iter())
                {
                    let hit = TriangleIntersector::intersect(
                        obj,
                        &ray_hit.ray.origin,
                        &ray_hit.ray.direction,
                    );
                    let hit_mask = hit.simd_ge(ray_hit.ray.near) & hit.simd_lt(ray_hit.ray.far);
                    if hit_mask.any() {
                        ray_hit.ray.far = hit_mask.if_else(|| hit, || ray_hit.ray.far);
                        let hit_mask = hit_mask.bitmask();
                        for i in 0..T::LANES {
                            if hit_mask & 1 << i != 0 {
                                ray_hit.obj_idx[i] = Some(obj_idx as u32);
                            }
                        }
                        simd_ray.update_far(&ray_hit.ray.far);
                    }
                }
            }
        }
    }
}
