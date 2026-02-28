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

    pub fn sample_eval<R: Rng>(rng: &mut R) -> (Vec3, f32, Vec3) {
        let dir_in = Self::sample(rng);
        let pfd = Self::pdf(dir_in);
        let f = Self::eval();

        (dir_in, pfd, f)
    }

    #[must_use]
    pub fn pdf(dir_in: Vec3) -> f32 {
        dir_in.z / PI
    }
}
