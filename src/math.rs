use hybrid_array::{
    ArraySize,
    sizes::{U4, U8},
};
use nalgebra::{Matrix3, SimdBool, SimdValue, Vector2, Vector3, Vector4};
use simba::simd::{WideF32x4, WideF32x8};

pub type Vec2f = Vector2<f32>;
pub type Vec3f = Vector3<f32>;
pub type Vec4f = Vector4<f32>;

pub type Vec3x4f = Vector3<WideF32x4>;

pub type Mat3f = Matrix3<f32>;

pub trait SimdType {
    type Lanes: ArraySize;
}

impl SimdType for WideF32x4 {
    type Lanes = U4;
}

impl SimdType for WideF32x8 {
    type Lanes = U8;
}

pub trait SimdBoolSplat {
    fn splat(val: bool) -> Self;
}

impl<T> SimdBoolSplat for T
where
    T: SimdBool + SimdValue<Element = bool>,
{
    fn splat(val: bool) -> Self {
        Self::splat(val)
    }
}
