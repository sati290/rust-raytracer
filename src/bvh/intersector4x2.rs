use arrayvec::ArrayVec;
use wide::{CmpGe as _, CmpLe as _, CmpLt as _, f32x4};

use crate::{
    bvh::{node_intersector4x2::BvhNodeIntersector4x2, simd_ray::SimdRay4x2Interleaved, *},
    ray::{Ray4, RayHit4},
    trace_stats::TraceStats,
};

pub struct BvhIntersector4x2 {}

impl BvhIntersector4x2 {
    #[must_use]
    pub fn occluded(bvh: &Bvh, ray: &Ray4, stats: &mut TraceStats) -> f32x4 {
        stats.trace_start(ray.valid.move_mask().count_ones() as u64);

        let mut active = ray.valid;
        let mut stack = ArrayVec::<_, 32>::new();
        let simd_ray = SimdRay4x2Interleaved::from(ray);

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

                let (hit_mask_l, hit_mask_r, _, _) =
                    BvhNodeIntersector4x2::intersect(child_bbox, &simd_ray);

                let hit_l = hit_mask_l & active;
                let hit_r = hit_mask_r & active;

                if hit_l.any() {
                    cur_node = Some(&children[0]);

                    if hit_r.any() {
                        stack.push((&children[1], hit_r));
                    }
                } else if hit_r.any() {
                    cur_node = Some(&children[1]);
                } else {
                    cur_node = None;
                }
            }

            if let Some(BvhNode::Leaf { triangles_range }) = cur_node {
                stats.leaf_visit(num_valid_rays as u64, triangles_range.len() as u64);
                for obj_idx in triangles_range.clone() {
                    let obj = &bvh.triangles[obj_idx];
                    let hit = obj.intersect_simd(&ray.origin, &ray.direction);
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

    pub fn intersect(bvh: &Bvh, ray_hit: &mut RayHit4, stats: &mut TraceStats) {
        stats.trace_start(ray_hit.ray.valid.move_mask().count_ones() as u64);

        let mut stack = ArrayVec::<_, 32>::new();
        let mut simd_ray = SimdRay4x2Interleaved::from(&ray_hit.ray);

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
                let (hit_l, hit_r, near_l, near_r) =
                    BvhNodeIntersector4x2::intersect(child_bbox, &simd_ray);

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
            }
        }
    }
}
