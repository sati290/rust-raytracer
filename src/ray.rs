use hybrid_array::Array;
use nalgebra::{SimdBool, SimdRealField, Vector3};

use crate::math::{SimdType, Vec3f, Vec4f};

#[derive(Clone)]
#[repr(C, align(16))]
pub struct StreamRay {
    pub origin_far: Vec4f,           // x, y, z, near
    pub direction_recip_near: Vec4f, // x, y, z, far
    pub direction: Vec4f,
}

impl StreamRay {
    #[must_use]
    #[inline]
    pub fn new(origin: &Vec3f, direction: &Vec3f, near: f32, far: f32) -> Self {
        let dir_recip = Vec3f::from_element(1.).component_div(direction);
        StreamRay {
            origin_far: Vec4f::new(origin.x, origin.y, origin.z, far),
            direction: Vec4f::new(direction.x, direction.y, direction.z, 0.),
            direction_recip_near: Vec4f::new(dir_recip.x, dir_recip.y, dir_recip.z, near),
        }
    }

    #[must_use]
    #[inline]
    pub fn _is_hit(&self) -> bool {
        self.origin_far.w < f32::INFINITY
    }

    #[must_use]
    #[inline]
    pub fn hit_dist(&self) -> f32 {
        self.origin_far.w
    }

    #[must_use]
    #[inline]
    pub fn hit_pos(&self) -> Vec3f {
        self.origin_far.xyz() + self.direction.xyz() * self.hit_dist()
    }
}

#[derive(Clone)]
pub struct Ray<T>
where
    T: SimdRealField,
{
    pub origin: Vector3<T>,
    pub direction: Vector3<T>,
    pub near: T,
    pub far: T,
    pub valid: T::SimdBool,
}

impl<T> Ray<T>
where
    T: SimdRealField<Element = f32> + Copy,
{
    #[must_use]
    #[inline]
    pub fn new(
        origin: &Vector3<T>,
        direction: &Vector3<T>,
        near: &T,
        far: &T,
        valid: &T::SimdBool,
    ) -> Self {
        Self {
            origin: *origin,
            direction: *direction,
            near: valid.if_else(|| *near, || T::splat(f32::INFINITY)),
            far: valid.if_else(|| *far, || T::splat(f32::NEG_INFINITY)),
            valid: *valid,
        }
    }

    #[must_use]
    #[inline]
    pub fn hit_dist(&self) -> T {
        self.far
    }

    #[must_use]
    #[inline]
    pub fn hit_pos(&self) -> Vector3<T> {
        self.origin + self.direction * self.hit_dist()
    }
}

#[derive(Clone)]
pub struct RayHit<T>
where
    T: SimdRealField + SimdType,
{
    pub ray: Ray<T>,
    pub obj_idx: Array<Option<u32>, T::Lanes>,
}

impl<T> From<Ray<T>> for RayHit<T>
where
    T: SimdRealField + SimdType,
{
    #[inline]
    fn from(ray: Ray<T>) -> Self {
        Self {
            ray,
            obj_idx: Default::default(),
        }
    }
}
