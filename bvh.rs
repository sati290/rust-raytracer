use crate::aabb::Aabb;
use crate::Vec3;
use crate::Vec3x4;
use crate::{Frustum, Ray, TraceResult, Triangle};
use arrayvec::ArrayVec;
use wide::CmpLt;

enum BvhNode {
    Inner {
        child_bbox: [Aabb; 2],
        children: [Box<BvhNode>; 2],
    },
    Leaf {
        object: usize,
    },
}

impl BvhNode {
    fn intersect_packet(
        bbox: &Aabb,
        rays: &[Ray],
        first_active_ray: usize,
        frustum: &Frustum,
    ) -> Option<usize> {
        let rays = &rays[first_active_ray..];
        if bbox.intersect(&rays[0].origin, &rays[0].direction_recip) {
            return Some(first_active_ray);
        }

        if bbox.intersect_frustum(frustum) {
            for (i, c) in rays.chunks_exact(4).enumerate() {
                let origins = Vec3x4::from([c[0].origin, c[1].origin, c[2].origin, c[3].origin]);
                let directions_recip = Vec3x4::from([
                    c[0].direction_recip,
                    c[1].direction_recip,
                    c[2].direction_recip,
                    c[3].direction_recip,
                ]);
                if bbox.intersect_simd(&origins, &directions_recip).any() {
                    return Some(first_active_ray + i * 4);
                }
            }

            for (i, r) in rays.chunks_exact(4).remainder().iter().enumerate() {
                if bbox.intersect(&r.origin, &r.direction_recip) {
                    return Some(first_active_ray + rays.len() / 4 * 4 + i);
                }
            }
        }

        None
    }
}

pub struct Bvh<'a> {
    objects: &'a [Triangle],
    root_node: BvhNode,
}

impl Bvh<'_> {
    pub fn build(objects: &[Triangle]) -> Bvh {
        let mut indices: Vec<_> = (0..objects.len()).collect();
        let object_bounds: Vec<_> = objects.iter().map(|o| (o.centroid(), o.aabb())).collect();
        let root_node = Bvh::build_recursive(&mut indices, &object_bounds);

        Bvh { objects, root_node }
    }

    fn build_recursive(indices: &mut [usize], object_bounds: &[(Vec3, Aabb)]) -> BvhNode {
        match indices.len() {
            1 => BvhNode::Leaf { object: indices[0] },
            2 => BvhNode::Inner {
                child_bbox: [object_bounds[indices[0]].1, object_bounds[indices[1]].1],
                children: [
                    Box::new(BvhNode::Leaf { object: indices[0] }),
                    Box::new(BvhNode::Leaf { object: indices[1] }),
                ],
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

                let child_left = Bvh::build_recursive(indices_left, object_bounds);
                let child_right = Bvh::build_recursive(indices_right, object_bounds);

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
                BvhNode::Leaf { object } => {
                    let hit =
                        self.objects[*object].intersect_simd::<false>(ray_origin, ray_direction);
                    result |= hit.cmp_lt(f32::INFINITY).move_mask();
                    if result == 0b1111 {
                        return result;
                    }
                }
            }
        }

        result
    }

    pub fn trace_packet<'a>(
        &'a self,
        rays: &[Ray],
        frustum: &Frustum,
        results: &mut [TraceResult<'a>],
    ) {
        let mut stack = ArrayVec::<_, 32>::new();

        stack.push((&self.root_node, 0));
        while let Some((node, first_active_ray)) = stack.pop() {
            match node {
                BvhNode::Inner {
                    child_bbox,
                    children,
                } => {
                    if let Some(i) =
                        BvhNode::intersect_packet(&child_bbox[0], rays, first_active_ray, frustum)
                    {
                        stack.push((&children[0], i))
                    };
                    if let Some(i) =
                        BvhNode::intersect_packet(&child_bbox[1], rays, first_active_ray, frustum)
                    {
                        stack.push((&children[1], i))
                    };
                }
                BvhNode::Leaf { object } => {
                    for (ray, result) in rays[first_active_ray..]
                        .iter()
                        .zip(&mut results[first_active_ray..])
                    {
                        let hit =
                            self.objects[*object].intersect::<false>(&ray.origin, &ray.direction);
                        if let Some(hit) = hit {
                            result.add_hit(hit, &self.objects[*object]);
                        }
                    }
                }
            }
        }
    }
}
