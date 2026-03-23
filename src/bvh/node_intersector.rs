use nalgebra::SimdRealField;
use simba::simd::{WideF32x4, WideF32x8};

use crate::{
    bvh::{
        node_intersector4::BvhNodeIntersector4, node_intersector8::BvhNodeIntersector8,
        simd_ray::SimdRay,
    },
    math::Vec3x4f,
};

pub trait BvhNodeIntersector<T: SimdRealField> {
    type SimdRay: SimdRay<T>;

    fn intersect(child_bbox: &Vec3x4f, index: usize, ray: &Self::SimdRay) -> (T::SimdBool, T);
}

pub trait BvhNodeIntersectorType: SimdRealField {
    type BvhNodeIntersector: BvhNodeIntersector<Self>;
}

impl BvhNodeIntersectorType for WideF32x4 {
    type BvhNodeIntersector = BvhNodeIntersector4;
}

impl BvhNodeIntersectorType for WideF32x8 {
    type BvhNodeIntersector = BvhNodeIntersector8;
}
