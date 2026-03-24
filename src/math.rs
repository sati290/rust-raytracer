use hybrid_array::{
    ArraySize,
    sizes::{U4, U8},
};
use nalgebra::{
    Matrix3, Quaternion, SimdBool, SimdRealField, SimdValue, UnitQuaternion, Vector2, Vector3,
    Vector4,
};
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
    #[must_use]
    fn splat(val: bool) -> Self;
}

impl<T> SimdBoolSplat for T
where
    T: SimdBool + SimdValue<Element = bool>,
{
    #[inline]
    fn splat(val: bool) -> Self {
        Self::splat(val)
    }
}

#[must_use]
#[inline]
pub fn fast_rotation_between<T>(from: &Vector3<T>, to: &Vector3<T>) -> UnitQuaternion<T>
where
    T: SimdRealField<Element: SimdRealField>,
{
    let dot = from.dot(to);
    let opposite_mask = dot.clone().simd_eq(-T::one());
    opposite_mask.if_else(
        || UnitQuaternion::from_scaled_axis(Vector3::<T>::y() * T::simd_pi()),
        || UnitQuaternion::new_normalize(Quaternion::from_parts(T::one() + dot, from.cross(to))),
    )
}
