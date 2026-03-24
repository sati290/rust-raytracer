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
        while let Some((node, node_active)) = stack.pop() {
            if (node_active & active).none() {
                continue;
            }

            let num_valid_rays = node_active.bitmask().count_ones();
            let mut cur_node = Some(node);
            while let Some(BvhNode::Inner {
                child_bbox,
                children,
            }) = cur_node
            {
                stats.inner_visit(num_valid_rays as u64);

                let mut hits = ArrayVec::<_, 2>::new();
                for (i, c) in children.iter().enumerate().take(2) {
                    let (hit, _) = T::BvhNodeIntersector::intersect(child_bbox, i, &simd_ray);

                    if hit.any() {
                        hits.push((c.as_ref(), hit));
                    }
                }

                if let Some(hit) = hits.pop() {
                    cur_node = Some(hit.0);
                    if let Some(hit) = hits.pop() {
                        stack.push(hit);
                    }
                } else {
                    cur_node = None;
                }
            }

            if let Some(BvhNode::Leaf { triangles_range }) = cur_node {
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

        let mut stack = ArrayVec::<_, 32>::new();
        let mut simd_ray = <<T as BvhNodeIntersectorType>::BvhNodeIntersector as BvhNodeIntersector<
            T,
        >>::SimdRay::from(&ray_hit.ray);

        stack.push((&bvh.root_node, ray_hit.ray.near));
        while let Some((node, node_near)) = stack.pop() {
            let node_ray_valid = node_near.simd_le(ray_hit.ray.far);
            if node_ray_valid.none() {
                continue;
            }

            let num_valid_rays = node_ray_valid.bitmask().count_ones();
            let mut cur_node = Some(node);
            while let Some(BvhNode::Inner {
                child_bbox,
                children,
            }) = cur_node
            {
                stats.inner_visit(num_valid_rays as u64);

                let mut hits = ArrayVec::<_, 2>::new();
                for (i, c) in children.iter().enumerate().take(2) {
                    let (hit, near) = T::BvhNodeIntersector::intersect(child_bbox, i, &simd_ray);

                    if hit.any() {
                        let dist = hit.if_else(|| near, || T::splat(f32::INFINITY));
                        hits.push((c.as_ref(), dist));
                    }
                }

                if hits.len() >= 2 && hits[0].1.simd_lt(hits[1].1).any() {
                    hits.swap(0, 1);
                }

                if let Some(hit) = hits.pop() {
                    cur_node = Some(hit.0);
                    if let Some(hit) = hits.pop() {
                        stack.push(hit);
                    }
                } else {
                    cur_node = None;
                }
            }

            if let Some(BvhNode::Leaf { triangles_range }) = cur_node {
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
                    ray_hit.ray.far = hit_mask.if_else(|| hit, || ray_hit.ray.far);
                    if hit_mask.any() {
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
