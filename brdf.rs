use std::f32::consts::PI;

use rand::{Rng, RngExt};
use ultraviolet::Vec3;

fn sample_cosine_weighted_hemisphere<R: Rng>(rng: &mut R) -> Vec3 {
    let u: [f32; 2] = rng.random();

    let a = u[0].sqrt();
    let b = 2. * PI * u[1];

    Vec3::new(a * b.cos(), a * b.sin(), (1. - u[0]).sqrt())
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
