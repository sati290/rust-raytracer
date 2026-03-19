mod build;
mod intersect1;
mod intersect1_simple;
mod intersect_stream;
mod intersector_n;
mod node_intersector4;
mod node_intersector8;
mod simd_ray;

use std::ops::Range;
use ultraviolet::Vec3x4;

use crate::triangle_opt::TriangleOpt;

pub use intersector_n::BvhIntersector4;
pub use intersector_n::BvhIntersector8;

enum BvhNode {
    Inner {
        // minl, minr, maxl, maxr
        child_bbox: Vec3x4,
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
