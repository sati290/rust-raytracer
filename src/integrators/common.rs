use std::marker::PhantomData;

use nalgebra::{SimdBool, SimdRealField, SimdValue, Vector3};
use rand::{Rng, distributions::Standard, prelude::Distribution};

use crate::{
    brdf::{Brdf, Brdf1},
    light::PointLight,
    math::{Vec3f, fast_rotation_between},
};

pub const SHADOW_RAY_NEAR: f32 = 1e-5;

#[derive(Clone)]
pub struct PathInfo {
    pub weight: Vec3f,
    pub destination_idx: u32,
    pub bounces: u8,
}

impl PathInfo {
    #[inline]
    pub fn diffuse(&self, weight: &Vec3f) -> Self {
        PathInfo {
            weight: self.weight.component_mul(weight),
            bounces: self.bounces + 1,
            ..*self
        }
    }
}

pub struct IntegratorsCommon1;

impl IntegratorsCommon1 {
    #[must_use]
    #[inline]
    pub fn sample_light(
        dir_out: &Vec3f,
        normal: &Vec3f,
        hit_pos: &Vec3f,
        light: &PointLight,
    ) -> Option<(Vec3f, f32, Vec3f)> {
        let light_vec = light.pos - *hit_pos;
        let ndotl = normal.dot(&light_vec);
        if ndotl * normal.dot(dir_out) < 0. {
            return None;
        }

        let light_dist_sq = light_vec.norm_squared();
        let light_dist = light_dist_sq.sqrt();
        let dir_in = light_vec / light_dist;
        let ndotl = (ndotl / light_dist).abs();
        let weight = light.intensity.component_mul(&Brdf1::eval()) * ndotl / light_dist_sq;

        if weight != Vec3f::zeros() {
            Some((dir_in, light_dist, weight))
        } else {
            None
        }
    }

    #[must_use]
    #[inline]
    pub fn sample_diffuse_ray<R: Rng>(
        dir_out: &Vec3f,
        normal: &Vec3f,
        rng: &mut R,
    ) -> (Vec3f, Vec3f) {
        let world_to_local = fast_rotation_between(normal, &Vec3f::z_axis());
        let dir_out_local = world_to_local * *dir_out;

        let (dir_in_local, pdf, brdf) = Brdf1::sample_eval(&dir_out_local, rng);
        let dir_in = world_to_local.inverse_transform_vector(&dir_in_local);
        let ndotl = normal.dot(&dir_in).abs();
        let weight = brdf * ndotl / pdf;

        (dir_in, weight)
    }
}

pub struct IntegratorsCommon<T> {
    _phantom: PhantomData<T>,
}

impl<T> IntegratorsCommon<T>
where
    T: SimdRealField<Element = f32> + Copy,
    Standard: Distribution<T>,
{
    #[must_use]
    #[inline]
    pub fn sample_light(
        dir_out: &Vector3<T>,
        normal: &Vector3<T>,
        hit_pos: &Vector3<T>,
        valid: &T::SimdBool,
        light: &PointLight,
    ) -> (Vector3<T>, T, Vector3<T>, T::SimdBool) {
        let light_vec = Vector3::<T>::splat(light.pos) - *hit_pos;
        let ndotl = normal.dot(&light_vec);
        let mut mask = (ndotl * normal.dot(dir_out)).simd_gt(T::zero()) & *valid;
        if mask.none() {
            return (
                Vector3::<T>::zeros(),
                T::zero(),
                Vector3::<T>::zeros(),
                mask,
            );
        }

        let light_dist_sq = light_vec.norm_squared();
        let light_dist = light_dist_sq.simd_sqrt();
        let dir_in = light_vec / light_dist;
        let ndotl = (ndotl / light_dist).simd_abs();
        let weight = Vector3::<T>::splat(light.intensity).component_mul(&Brdf::<T>::eval()) * ndotl
            / light_dist_sq;

        mask = mask & weight.norm_squared().simd_gt(T::zero());

        (dir_in, light_dist, weight, mask)
    }

    #[must_use]
    #[inline]
    pub fn sample_diffuse_ray<R: Rng>(
        dir_out: &Vector3<T>,
        normal: &Vector3<T>,
        rng: &mut R,
    ) -> (Vector3<T>, Vector3<T>) {
        let world_to_local = fast_rotation_between(normal, &Vector3::<T>::z_axis());

        let dir_out_local = world_to_local * *dir_out;
        let (dir_in_local, pdf, brdf) = Brdf::<T>::sample_eval(&dir_out_local, rng);
        let dir_in = world_to_local.inverse_transform_vector(&dir_in_local);
        let ndotl = normal.dot(&dir_in).simd_abs();
        let weight = brdf * ndotl / pdf;

        (dir_in, weight)
    }
}
