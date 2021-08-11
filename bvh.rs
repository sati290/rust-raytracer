use crate::aabb::Aabb;
use crate::Sphere;
use std::cmp::Ordering;
use std::ops::Range;
use ultraviolet::Vec3;

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
        ray_origin: &Vec3,
        ray_direction: &Vec3,
        ray_direction_recip: &Vec3,
    ) -> (f32, Option<&Sphere>) {
        match self {
            BvhNode::Inner {
                child_bbox,
                children,
            } => {
                let tl = if child_bbox[0].intersect(ray_origin, ray_direction_recip) {
                    children[0].trace(ray_origin, ray_direction, ray_direction_recip)
                } else {
                    (f32::MAX, None)
                };
                let tr = if child_bbox[1].intersect(ray_origin, ray_direction_recip) {
                    children[1].trace(ray_origin, ray_direction, ray_direction_recip)
                } else {
                    (f32::MAX, None)
                };

                if tl.0 < tr.0 {
                    tl
                } else {
                    tr
                }
            }
            BvhNode::Leaf { object } => match object.intersect(ray_origin, ray_direction, false) {
                Some(x) => (x, Some(object)),
                None => (f32::MAX, None),
            },
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

    pub fn trace(&self, ray_origin: &Vec3, ray_direction: &Vec3) -> (f32, Option<&Sphere>) {
        let ray_direction_recip = Vec3::broadcast(1.) / *ray_direction;
        self.root_node
            .trace(ray_origin, ray_direction, &ray_direction_recip)
    }
}
