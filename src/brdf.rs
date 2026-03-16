use std::f32::consts::PI;

use rand::{Rng, RngExt};
use ultraviolet::{Vec3, Vec3x4};
use wide::f32x4;

fn sample_cosine_weighted_hemisphere<R: Rng>(rng: &mut R) -> Vec3 {
    let u: [f32; 2] = rng.random();

    let a = u[0].sqrt();
    let b = 2. * PI * u[1];

    Vec3::new(a * b.cos(), a * b.sin(), (1. - u[0]).sqrt())
}

fn sample_cosine_weighted_hemisphere4<R: Rng>(rng: &mut R) -> Vec3x4 {
    let u0 = f32x4::from(rng.random::<[f32; 4]>());
    let u1 = f32x4::from(rng.random::<[f32; 4]>());

    let a = u0.sqrt();
    let b = 2. * f32x4::PI * u1;

    Vec3x4::new(a * b.cos(), a * b.sin(), (f32x4::ONE - u0).sqrt())
}

pub struct Brdf {}

impl Brdf {
    #[must_use]
    pub fn eval() -> Vec3 {
        Vec3::broadcast(0.8 / PI)
    }

    #[must_use]
    pub fn sample<R: Rng>(rng: &mut R) -> Vec3 {
        sample_cosine_weighted_hemisphere(rng)
    }

    #[must_use]
    pub fn sample_eval<R: Rng>(dir_out: &Vec3, rng: &mut R) -> (Vec3, f32, Vec3) {
        let mut dir_in = Self::sample(rng);
        let pfd = dir_in.z / PI;
        let f = Self::eval();

        dir_in.z = dir_in.z.copysign(dir_out.z);

        (dir_in, pfd, f)
    }

    #[must_use]
    pub fn _pdf(dir_out: Vec3, dir_in: Vec3) -> f32 {
        if dir_out.z * dir_in.z > 0. {
            dir_in.z / PI
        } else {
            0.
        }
    }
}

pub struct Brdf4 {}

impl Brdf4 {
    #[must_use]
    pub fn eval() -> Vec3x4 {
        Vec3x4::broadcast(f32x4::splat(0.8 / PI))
    }

    #[must_use]
    pub fn sample<R: Rng>(rng: &mut R) -> Vec3x4 {
        sample_cosine_weighted_hemisphere4(rng)
    }

    #[must_use]
    pub fn sample_eval<R: Rng>(dir_out: &Vec3x4, rng: &mut R) -> (Vec3x4, f32x4, Vec3x4) {
        let mut dir_in = Self::sample(rng);
        let pfd = dir_in.z / PI;
        let f = Self::eval();

        dir_in.z = dir_in.z.copysign(dir_out.z);

        (dir_in, pfd, f)
    }
}
