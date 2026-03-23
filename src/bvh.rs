mod build;
mod intersect1;
mod intersect1_simple;
mod intersect_stream;
mod intersector_n;
mod node_intersector;
mod node_intersector4;
mod node_intersector8;
mod simd_ray;

use std::ops::Range;

use crate::math::Vec3x4f;
use crate::triangle_opt::TriangleOpt;

pub use intersector_n::BvhIntersector;
pub use node_intersector::BvhNodeIntersectorType;

enum BvhNode {
    Inner {
        // minl, minr, maxl, maxr
        child_bbox: Vec3x4f,
        children: [Box<BvhNode>; 2],
    },
    Leaf {
        triangles_range: Range<usize>,
    },
}

pub struct Bvh {
    triangles: Vec<TriangleOpt>,
    object_indices: Vec<usize>,
    root_node: BvhNode,
}
