use crate::aabb::Aabb;
use crate::f32x4;
use crate::Sphere;
use crate::Vec3;
use crate::Vec3x4;
use std::cmp::Ordering;
use std::ops::Range;

enum BvhNode<'a> {
    Inner {
        child_bbox: [Aabb; 2],
        children: [Box<BvhNode<'a>>; 2],
    },
    Leaf {
        object: &'a Sphere,
    },
}

impl BvhNode<'_> {
    fn trace(
        &self,
        ray_origin: &Vec3x4,
        ray_direction: &Vec3x4,
        ray_direction_recip: &Vec3x4,
    ) -> (f32x4, [Option<&Sphere>; 4]) {
        match self {
            BvhNode::Inner {
                child_bbox,
                children,
            } => {
                let tl = if child_bbox[0]
                    .intersect_simd(ray_origin, ray_direction_recip)
                    .any()
                {
                    children[0].trace(ray_origin, ray_direction, ray_direction_recip)
                } else {
                    (f32x4::splat(f32::MAX), [None; 4])
                };
                let tr = if child_bbox[1]
                    .intersect_simd(ray_origin, ray_direction_recip)
                    .any()
                {
                    children[1].trace(ray_origin, ray_direction, ray_direction_recip)
                } else {
                    (f32x4::splat(f32::MAX), [None; 4])
                };

                if tl.0 < tr.0 {
                    tl
                } else {
                    tr
                }
            }
            BvhNode::Leaf { object } => {
                match object.intersect_simd(ray_origin, ray_direction, false) {
                    Some(x) => (x, Some(object)),
                    None => (f32::MAX, None),
                }
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
        let mut bounds = Aabb::empty();
        let mut centroid_bounds = Aabb::empty();
        for o in &*objects {
            bounds.join_mut(o.aabb());
            centroid_bounds.grow_mut(o.center);
        }

        if objects.len() > 1 {
            let size = bounds.size();
            let largest_axis = if size.x > size.y && size.x > size.z {
                0
            } else if size.y > size.z {
                1
            } else {
                2
            };

            objects.sort_by(|a, b| {
                a.center[largest_axis]
                    .partial_cmp(&b.center[largest_axis])
                    .unwrap_or(Ordering::Equal)
            });

            let (objects_left, objects_right) = objects.split_at_mut(objects.len() / 2);

            let mut bbox_left = Aabb::empty();
            for o in &*objects_left {
                bbox_left.join_mut(o.aabb());
            }

            let mut bbox_right = Aabb::empty();
            for o in &*objects_right {
                bbox_right.join_mut(o.aabb());
            }

            let child_left = Bvh::build_recursive(objects_left);
            let child_right = Bvh::build_recursive(objects_right);

            BvhNode::Inner {
                child_bbox: [bbox_left, bbox_right],
                children: [Box::new(child_left), Box::new(child_right)],
            }
        } else {
            BvhNode::Leaf {
                object: objects.first().unwrap(),
            }
        }
    }

    pub fn trace(
        &self,
        ray_origin: &Vec3x4,
        ray_direction: &Vec3x4,
    ) -> (f32x4, [Option<&Sphere>; 4]) {
        let ray_direction_recip = Vec3x4::splat(Vec3::broadcast(1.)) / *ray_direction;
        self.root_node
            .trace(ray_origin, ray_direction, &ray_direction_recip)
    }
}
