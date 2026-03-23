use arrayvec::ArrayVec;
use nalgebra::SimdValue;

use crate::{aabb::Aabb, bvh::*, ray::StreamRay, trace_stats::TraceStats};

impl Bvh {
    pub fn _occluded1_simple(&self, ray: &StreamRay, stats: &mut TraceStats) -> bool {
        stats.trace_start(1);

        let mut stack = ArrayVec::<&BvhNode, 32>::new();
        stack.push(&self.root_node);

        while let Some(node) = stack.pop() {
            match node {
                BvhNode::Inner {
                    child_bbox,
                    children,
                } => {
                    stats.inner_visit(1);

                    let bb_l = Aabb {
                        min: child_bbox.extract(0),
                        max: child_bbox.extract(2),
                    };
                    let bb_r = Aabb {
                        min: child_bbox.extract(1),
                        max: child_bbox.extract(3),
                    };

                    if bb_l._intersect(&ray.origin_far.xyz(), &ray.direction_recip_near.xyz()) {
                        stack.push(&children[0]);
                    }

                    if bb_r._intersect(&ray.origin_far.xyz(), &ray.direction_recip_near.xyz()) {
                        stack.push(&children[1]);
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
                }
            }
        }

        false
    }
}
