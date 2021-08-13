use crate::aabb::{Aabb, AabbSimdx2};
use crate::Sphere;
use crate::TraceResultSimd;
use ultraviolet::{Vec3, Vec3x4, Vec3x8};

enum BvhNode<'a> {
    Inner {
        child_bounds: AabbSimdx2,
        children: [Box<BvhNode<'a>>; 2],
    },
    Leaf {
        object: &'a Sphere,
    },
}

struct BvhNodeStack<'a> {
    stack: [Option<&'a BvhNode<'a>>; 32],
    size: usize,
}

impl<'a> BvhNodeStack<'a> {
    fn new() -> Self {
        BvhNodeStack {
            stack: Default::default(),
            size: 0,
        }
    }

    fn len(&self) -> usize {
        self.size
    }

    fn push(&mut self, node: &'a BvhNode) {
        self.stack[self.size] = Some(node);
        self.size += 1;
    }

    fn pop(&mut self) -> &'a BvhNode<'a> {
        self.size -= 1;
        self.stack[self.size].unwrap()
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
                child_bounds: AabbSimdx2::from([bounds_l, bounds_r]),
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

        let ray_originx8: [Vec3; 4] = (*ray_origin).into();
        let ray_originx8 = Vec3x8::from([
            ray_originx8[0],
            ray_originx8[1],
            ray_originx8[2],
            ray_originx8[3],
            ray_originx8[0],
            ray_originx8[1],
            ray_originx8[2],
            ray_originx8[3],
        ]);
        let ray_direction_recipx8: [Vec3; 4] = ray_direction_recip.into();
        let ray_direction_recipx8 = Vec3x8::from([
            ray_direction_recipx8[0],
            ray_direction_recipx8[1],
            ray_direction_recipx8[2],
            ray_direction_recipx8[3],
            ray_direction_recipx8[0],
            ray_direction_recipx8[1],
            ray_direction_recipx8[2],
            ray_direction_recipx8[3],
        ]);

        let mut result = TraceResultSimd::new();
        let mut stack = BvhNodeStack::new();

        stack.push(&self.root_node);
        while stack.len() > 0 {
            let node = stack.pop();
            match node {
                BvhNode::Inner {
                    child_bounds,
                    children,
                } => {
                    let aabb_hit = child_bounds.intersect(&ray_originx8, &ray_direction_recipx8);
                    let hit_mask = aabb_hit.move_mask();

                    if hit_mask & 0b00001111 != 0 {
                        stack.push(&children[0]);
                    }
                    if hit_mask & 0b11110000 != 0 {
                        stack.push(&children[1]);
                    }
                }
                BvhNode::Leaf { object } => {
                    let hit = object.intersect_simd(ray_origin, ray_direction, false);
                    result.add_hit(hit, object);
                }
            }
        }

        result
    }
}
