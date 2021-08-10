use crate::aabb::Aabb;
use crate::Sphere;
use std::cmp::Ordering;
use std::ops::Range;

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
    objects: Vec<&'a Sphere>,
    root_node: BvhNode,
}

impl Bvh<'_> {
    pub fn build(objects: &[Sphere]) -> Bvh {
        let mut objects: Vec<&Sphere> = objects.iter().collect();
        let root_node = Bvh::build_recursive(&mut objects);

        Bvh { objects, root_node }
    }

    fn build_recursive(objects: &mut [&Sphere]) -> BvhNode {
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

            let child_left = Bvh::build_recursive(objects_left);
            let child_right = Bvh::build_recursive(objects_right);

            BvhNode::Leaf { objects: 0..0 }
        } else {
            BvhNode::Leaf { objects: 0..0 }
        }
    }
}
