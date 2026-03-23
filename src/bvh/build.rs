use std::ops::{Add, AddAssign};

use crate::{aabb::Aabb, bvh::*, math::Vec3f, mesh::TriangleMesh, triangle_opt::TriangleOpt};

#[derive(Default, Debug)]
struct BvhStats {
    num_leaves: u32,
    num_objs: u32,
    max_depth: u32,
    max_objs: u32,
}

impl BvhStats {
    fn new() -> Self {
        Default::default()
    }

    fn add_leaf(&mut self, depth: u32, objs: u32) {
        self.num_leaves += 1;
        self.num_objs += objs;
        self.max_depth = self.max_depth.max(depth);
        self.max_objs = self.max_objs.max(objs);
    }
}

impl Add for BvhStats {
    type Output = BvhStats;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl AddAssign for BvhStats {
    fn add_assign(&mut self, rhs: Self) {
        self.num_leaves += rhs.num_leaves;
        self.num_objs += rhs.num_objs;
        self.max_depth = self.max_depth.max(rhs.max_depth);
        self.max_objs = self.max_objs.max(rhs.max_objs);
    }
}

impl Bvh {
    pub fn build(mesh: &TriangleMesh) -> Bvh {
        let mut object_indices: Vec<_> = (0..mesh.num_triangles()).collect();
        let object_bounds: Vec<_> = mesh.iter().map(|t| (t.centroid(), t.aabb())).collect();

        let mut stats = BvhStats::new();
        let root_node = Bvh::build_recursive(&mut object_indices, 0, &object_bounds, 0, &mut stats);
        println!("{:?}", stats);

        let triangles = object_indices
            .iter()
            .map(|&index| TriangleOpt::from(&mesh.get_triangle(index as u32)))
            .collect();

        Bvh {
            triangles,
            object_indices,
            root_node,
        }
    }

    fn build_recursive(
        indices: &mut [usize],
        indices_start_idx: usize,
        object_bounds: &[(Vec3f, Aabb)],
        depth: u32,
        stats: &mut BvhStats,
    ) -> BvhNode {
        let mut leaf = || {
            stats.add_leaf(depth, indices.len() as u32);

            BvhNode::Leaf {
                triangles_range: indices_start_idx..indices_start_idx + indices.len(),
            }
        };

        match indices.len() {
            1..=8 => leaf(),
            _ => {
                let mut bounds = Aabb::empty();
                let mut centroid_bounds = Aabb::empty();
                for idx in &*indices {
                    let (centroid, aabb) = &object_bounds[*idx];
                    bounds.join_mut(aabb);
                    centroid_bounds.grow_mut(centroid);
                }

                let size = centroid_bounds.size();

                if centroid_bounds.size().norm_squared() == 0. {
                    println!(
                        "centroid_bounds.size() == 0, adding leaf with {} objects, depth {}",
                        indices.len(),
                        depth
                    );

                    return leaf();
                }

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
                    |centroid: &Vec3f| (k1 * (centroid[largest_axis] - k0)) as usize;
                for idx in &*indices {
                    let (centroid, aabb) = &object_bounds[*idx];
                    let bucket_idx = get_bucket_idx(centroid);
                    let bucket = &mut buckets[bucket_idx];

                    bucket.0.join_mut(aabb);
                    bucket.1 += 1;
                }

                let mut min_bucket = 0;
                let mut min_cost = f32::INFINITY;
                let mut bounds_l = Aabb::empty();
                let mut bounds_r = Aabb::empty();
                for i in 0..(BUCKET_COUNT - 1) {
                    let (buckets_l, buckets_r) = buckets.split_at(i + 1);
                    let child_l = buckets_l.iter().fold((Aabb::empty(), 0u32), |acc, x| {
                        (acc.0.join(&x.0), acc.1 + x.1)
                    });
                    let child_r = buckets_r.iter().fold((Aabb::empty(), 0u32), |acc, x| {
                        (acc.0.join(&x.0), acc.1 + x.1)
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

                let mut stats_r = BvhStats::new();
                let (child_left, child_right) = rayon::join(
                    || {
                        Bvh::build_recursive(
                            indices_left,
                            indices_start_idx,
                            object_bounds,
                            depth + 1,
                            stats,
                        )
                    },
                    || {
                        Bvh::build_recursive(
                            indices_right,
                            indices_start_idx + split_idx,
                            object_bounds,
                            depth + 1,
                            &mut stats_r,
                        )
                    },
                );
                *stats += stats_r;

                BvhNode::Inner {
                    child_bbox: Vec3x4f::from([
                        bounds_l.min,
                        bounds_r.min,
                        bounds_l.max,
                        bounds_r.max,
                    ]),
                    children: [Box::new(child_left), Box::new(child_right)],
                }
            }
        }
    }
}
