use std::f32::consts::PI;

use rand::{Rng, RngExt};
use ultraviolet::{Vec3, Vec3x4, Vec3x8};
use wide::{f32x4, f32x8};

#[must_use]
#[inline]
fn sample_cosine_weighted_hemisphere<R: Rng>(rng: &mut R) -> Vec3 {
    let u: [f32; 2] = rng.random();

    let a = u[0].sqrt();
    let b = 2. * PI * u[1];

    Vec3::new(a * b.cos(), a * b.sin(), (1. - u[0]).sqrt())
}

pub struct Brdf1 {}

impl Brdf1 {
    #[must_use]
    #[inline]
    pub fn eval() -> Vec3 {
        Vec3::broadcast(0.8 / PI)
    }

    #[must_use]
    #[inline]
    pub fn sample<R: Rng>(rng: &mut R) -> Vec3 {
        sample_cosine_weighted_hemisphere(rng)
    }

    #[must_use]
    #[inline]
    pub fn sample_eval<R: Rng>(dir_out: &Vec3, rng: &mut R) -> (Vec3, f32, Vec3) {
        let mut dir_in = Self::sample(rng);
        let pfd = dir_in.z / PI;
        let f = Self::eval();

        dir_in.z = dir_in.z.copysign(dir_out.z);

        (dir_in, pfd, f)
    }

    #[must_use]
    #[inline]
    pub fn _pdf(dir_out: Vec3, dir_in: Vec3) -> f32 {
        if dir_out.z * dir_in.z > 0. {
            dir_in.z / PI
        } else {
            0.
        }
    }
}

macro_rules! brdf_n {
    ($(($n:ident, $t:ident, $vt:ident, $at:ty)),+) => {
        $(
            pub struct $n;

            impl $n {
                #[must_use]
                #[inline]
                pub fn eval() -> $vt {
                    $vt::broadcast($t::splat(0.8 / PI))
                }

                #[must_use]
                #[inline]
                pub fn sample<R: Rng>(rng: &mut R) -> $vt {
                    let u0 = $t::from(rng.random::<$at>());
                    let u1 = $t::from(rng.random::<$at>());

                    let a = u0.sqrt();
                    let b = 2. * $t::PI * u1;

                    $vt::new(a * b.cos(), a * b.sin(), ($t::ONE - u0).sqrt())
                }

                #[must_use]
                #[inline]
                pub fn sample_eval<R: Rng>(dir_out: &$vt, rng: &mut R) -> ($vt, $t, $vt) {
                    let mut dir_in = Self::sample(rng);
                    let pfd = dir_in.z / PI;
                    let f = Self::eval();

                    dir_in.z = dir_in.z.copysign(dir_out.z);

                    (dir_in, pfd, f)
                }
            }
        )+
    };
}

brdf_n!(
    (Brdf4, f32x4, Vec3x4, [f32; 4]),
    (Brdf8, f32x8, Vec3x8, [f32; 8])
);
