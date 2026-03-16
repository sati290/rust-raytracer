mod build;
mod intersect1;
mod intersect1_simple;
mod intersect_stream;
mod intersector4;
mod simd_ray;

use std::ops::Range;
use ultraviolet::Vec3x4;

use crate::triangle_opt::TriangleOpt;

pub use intersector4::BvhIntersector4;

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
