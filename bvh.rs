use crate::aabb::{Aabb, AabbSimd};
use crate::Sphere;
use crate::TraceResultSimd;
use crate::Vec3;
use crate::Vec3x4;

enum BvhNode<'a> {
    Inner {
        child_bbox: [AabbSimd; 2],
        children: [Box<BvhNode<'a>>; 2],
    },
    Leaf {
        object: &'a Sphere,
    },
}

impl<'a> BvhNode<'a> {
    fn trace(
        &self,
        ray_origin: &Vec3x4,
        ray_direction: &Vec3x4,
        ray_direction_recip: &Vec3x4,
        result: &mut TraceResultSimd<'a>,
    ) {
        match self {
            BvhNode::Inner {
                child_bbox,
                children,
            } => {
                if child_bbox[0]
                    .intersect(ray_origin, ray_direction_recip)
                    .any()
                {
                    children[0].trace(ray_origin, ray_direction, ray_direction_recip, result)
                };
                if child_bbox[1]
                    .intersect(ray_origin, ray_direction_recip)
                    .any()
                {
                    children[1].trace(ray_origin, ray_direction, ray_direction_recip, result)
                };
            }
            BvhNode::Leaf { object } => {
                let hit = object.intersect_simd(ray_origin, ray_direction, false);
                result.add_hit(hit, object);
            }
        }
    }
}

pub struct Bvh<'a> {
    objects: Vec<&'a Sphere>,
    root_node: BvhNode<'a>,
}

impl Bvh<'_> {
    pub fn build(objects: &[Sphere]) -> Bvh {
        let mut objects: Vec<&Sphere> = objects.iter().collect();
        let root_node = Bvh::build_recursive(&mut objects);

        Bvh { objects, root_node }
    }

    fn build_recursive<'a>(objects: &mut [&'a Sphere]) -> BvhNode<'a> {
        if objects.len() > 1 {
            let mut bounds = Aabb::empty();
            let mut centroid_bounds = Aabb::empty();
            for o in &*objects {
                bounds.join_mut(o.aabb());
                centroid_bounds.grow_mut(o.center);
            }

            let size = bounds.size();
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
            let get_bucket_idx = |obj: &Sphere| (k1 * (obj.center[largest_axis] - k0)) as usize;
            for o in &*objects {
                let bucket_idx = get_bucket_idx(o);
                let bucket = &mut buckets[bucket_idx];

                bucket.0.join_mut(o.aabb());
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

            objects.sort_by(|a, b| get_bucket_idx(a).cmp(&get_bucket_idx(b)));
            let split_idx = objects.partition_point(|o| get_bucket_idx(o) <= min_bucket);
            let (objects_left, objects_right) = objects.split_at_mut(split_idx);

            let child_left = Bvh::build_recursive(objects_left);
            let child_right = Bvh::build_recursive(objects_right);

            BvhNode::Inner {
                child_bbox: [AabbSimd::from(bounds_l), AabbSimd::from(bounds_r)],
                children: [Box::new(child_left), Box::new(child_right)],
            }
        } else {
            BvhNode::Leaf {
                object: objects.first().unwrap(),
            }
        }
    }

    pub fn trace(&self, ray_origin: &Vec3x4, ray_direction: &Vec3x4) -> TraceResultSimd {
        let ray_direction_recip = Vec3x4::splat(Vec3::broadcast(1.)) / *ray_direction;
        let mut result = TraceResultSimd::new();
        self.root_node
            .trace(ray_origin, ray_direction, &ray_direction_recip, &mut result);

        result
    }
}
