mod build;
mod intersect1;
mod intersect1_simple;
mod intersect4;
mod intersect_stream;

use std::ops::Range;
use ultraviolet::Vec3x4;

use crate::triangle_opt::TriangleOpt;

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
