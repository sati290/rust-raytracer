use arrayvec::ArrayVec;
use wide::{CmpGe as _, CmpLe as _, CmpLt as _, f32x4, f32x8};

use crate::{
    bvh::{
        node_intersector4::BvhNodeIntersector4,
        node_intersector8::BvhNodeIntersector8,
        simd_ray::{SimdRay4, SimdRay8},
        *,
    },
    ray::{Ray4, Ray8, RayHit4, RayHit8},
    trace_stats::TraceStats,
    triangle_intersector::{TriangleIntersector4, TriangleIntersector8},
};

macro_rules! bvh_intersector_n {
    ($(($name:ident, $n:literal, $scalar:ident, $vector:ident, $ray:ident, $rayhit:ident, $simdray:ident, $node_intersector:ident, $triangle_intersector:ident)),+) => {
        $(
            pub struct $name {}

            impl $name {
                #[must_use]
                pub fn occluded(bvh: &Bvh, ray: &$ray, stats: &mut TraceStats) -> $scalar {
                    stats.trace_start(ray.valid.move_mask().count_ones() as u64);

                    let mut active = ray.valid;
                    let mut stack = ArrayVec::<_, 32>::new();
                    let simd_ray = $simdray::from(ray);

                    stack.push((&bvh.root_node, active));
                    while let Some((node, node_active)) = stack.pop() {
                        if (node_active & active).none() {
                            continue;
                        }

                        let num_valid_rays = node_active.move_mask().count_ones();
                        let mut cur_node = Some(node);
                        while let Some(BvhNode::Inner {
                            child_bbox,
                            children,
                        }) = cur_node
                        {
                            stats.inner_visit(num_valid_rays as u64);

                            let mut hits = ArrayVec::<_, 2>::new();
                            for i in 0..2 {
                                let (hit, _) = $node_intersector::intersect(child_bbox, i, &simd_ray);

                                if hit.any() {
                                    hits.push((&*children[i], hit));
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
                                let hit = $triangle_intersector::intersect(obj, &ray.origin, &ray.direction);
                                let hit_mask = hit.cmp_ge(ray.near) & hit.cmp_lt(ray.far);
                                active &= !hit_mask;
                                if active.none() {
                                    return !active & ray.valid;
                                }
                            }
                        }
                    }

                    !active & ray.valid
                }

                pub fn intersect(bvh: &Bvh, ray_hit: &mut $rayhit, stats: &mut TraceStats) {
                    stats.trace_start(ray_hit.ray.valid.move_mask().count_ones() as u64);

                    let mut stack = ArrayVec::<_, 32>::new();
                    let mut simd_ray = $simdray::from(&ray_hit.ray);

                    stack.push((&bvh.root_node, ray_hit.ray.near));
                    while let Some((node, node_near)) = stack.pop() {
                        let node_ray_valid = node_near.cmp_le(ray_hit.ray.far);
                        if node_ray_valid.none() {
                            continue;
                        }

                        let num_valid_rays = node_ray_valid.move_mask().count_ones();
                        let mut cur_node = Some(node);
                        while let Some(BvhNode::Inner {
                            child_bbox,
                            children,
                        }) = cur_node
                        {
                            stats.inner_visit(num_valid_rays as u64);

                            let mut hits = ArrayVec::<_, 2>::new();
                            for i in 0..2 {
                                let (hit, near) = $node_intersector::intersect(child_bbox, i, &simd_ray);

                                if hit.any() {
                                    let dist = hit.blend(near, $scalar::splat(f32::INFINITY));
                                    hits.push((&*children[i], dist));
                                }
                            }

                            if hits.len() >= 2 && hits[0].1.cmp_lt(hits[1].1).any() {
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
                                let hit = $triangle_intersector::intersect(obj, &ray_hit.ray.origin, &ray_hit.ray.direction);
                                let hit_mask = (hit.cmp_ge(ray_hit.ray.near) & hit.cmp_lt(ray_hit.ray.far))
                                    .move_mask() as u32;
                                let hit: [f32; $n] = hit.into();
                                let far = ray_hit.ray.far.as_array_mut();
                                for i in 0..$n {
                                    if hit_mask & 1 << i != 0 {
                                        far[i] = hit[i];
                                        ray_hit.obj_idx[i] = Some(obj_idx as u32);
                                    }
                                }
                                if hit_mask != 0 {
                                    simd_ray.update_far(far);
                                }
                            }
                        }
                    }
                }
            }
        )+
    }
}

bvh_intersector_n!(
    (
        BvhIntersector4,
        4,
        f32x4,
        Vec3x4,
        Ray4,
        RayHit4,
        SimdRay4,
        BvhNodeIntersector4,
        TriangleIntersector4
    ),
    (
        BvhIntersector8,
        8,
        f32x8,
        Vec3x8,
        Ray8,
        RayHit8,
        SimdRay8,
        BvhNodeIntersector8,
        TriangleIntersector8
    )
);
