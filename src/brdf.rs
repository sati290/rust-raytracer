use std::{f32::consts::PI, marker::PhantomData};

use nalgebra::{SimdRealField, Vector3};
use rand::{Rng, distributions::Standard, prelude::Distribution};

use crate::math::Vec3f;

#[must_use]
#[inline]
fn sample_cosine_weighted_hemisphere<R: Rng>(rng: &mut R) -> Vec3f {
    let u: [f32; 2] = rng.r#gen();

    let a = u[0].sqrt();
    let b = 2. * PI * u[1];

    Vec3f::new(a * b.cos(), a * b.sin(), (1. - u[0]).sqrt())
}

pub struct Brdf1 {}

impl Brdf1 {
    #[must_use]
    #[inline]
    pub fn eval() -> Vec3f {
        Vec3f::from_element(0.8 / PI)
    }

    #[must_use]
    #[inline]
    pub fn sample<R: Rng>(rng: &mut R) -> Vec3f {
        sample_cosine_weighted_hemisphere(rng)
    }

    #[must_use]
    #[inline]
    pub fn sample_eval<R: Rng>(dir_out: &Vec3f, rng: &mut R) -> (Vec3f, f32, Vec3f) {
        let mut dir_in = Self::sample(rng);
        let pfd = dir_in.z / PI;
        let f = Self::eval();

        dir_in.z = dir_in.z.copysign(dir_out.z);

        (dir_in, pfd, f)
    }

    #[must_use]
    #[inline]
    pub fn _pdf(dir_out: Vec3f, dir_in: Vec3f) -> f32 {
        if dir_out.z * dir_in.z > 0. {
            dir_in.z / PI
        } else {
            0.
        }
    }
}

pub struct Brdf<T> {
    _phantom: PhantomData<T>,
}

impl<T> Brdf<T>
where
    T: SimdRealField<Element = f32> + Copy,
    Standard: Distribution<T>,
{
    #[must_use]
    #[inline]
    pub fn eval() -> Vector3<T> {
        Vector3::<T>::from_element(T::splat(0.8 / PI))
    }

    #[must_use]
    #[inline]
    pub fn sample<R: Rng>(rng: &mut R) -> Vector3<T> {
        let u0 = rng.r#gen::<T>();
        let u1 = rng.r#gen::<T>();

        let a = u0.simd_sqrt();
        let b = T::splat(2.) * T::simd_pi() * u1;

        Vector3::<T>::new(
            a * b.simd_cos(),
            a * b.simd_sin(),
            (T::one() - u0).simd_sqrt(),
        )
    }

    #[must_use]
    #[inline]
    pub fn sample_eval<R: Rng>(dir_out: &Vector3<T>, rng: &mut R) -> (Vector3<T>, T, Vector3<T>) {
        let mut dir_in = Self::sample(rng);
        let pfd = dir_in.z / T::simd_pi();
        let f = Self::eval();

        dir_in.z = dir_in.z.simd_copysign(dir_out.z);

        (dir_in, pfd, f)
    }
}
