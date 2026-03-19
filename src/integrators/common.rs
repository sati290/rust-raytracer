use std::f32::consts::PI;

use rand::Rng;
use ultraviolet::{Rotor3, Rotor3x4, Rotor3x8, Vec3, Vec3x4, Vec3x8};
use wide::{CmpEq as _, CmpGt as _, f32x4, f32x8};

use crate::{
    brdf::{Brdf1, Brdf4, Brdf8},
    light::PointLight,
    utils::BlendRotor,
};

pub const SHADOW_RAY_NEAR: f32 = 1e-5;

#[derive(Clone)]
pub struct PathInfo {
    pub weight: Vec3,
    pub destination_idx: u32,
    pub bounces: u8,
}

impl PathInfo {
    pub fn diffuse(&self, weight: &Vec3) -> Self {
        PathInfo {
            weight: self.weight * *weight,
            bounces: self.bounces + 1,
            ..*self
        }
    }
}

pub struct IntegratorsCommon1;

impl IntegratorsCommon1 {
    #[must_use]
    pub fn sample_light(
        dir_out: &Vec3,
        normal: &Vec3,
        hit_pos: &Vec3,
        light: &PointLight,
    ) -> Option<(Vec3, f32, Vec3)> {
        let light_vec = light.pos - *hit_pos;
        let ndotl = normal.dot(light_vec);
        if ndotl * normal.dot(*dir_out) < 0. {
            return None;
        }

        let light_dist_sq = light_vec.mag_sq();
        let light_dist = light_dist_sq.sqrt();
        let dir_in = light_vec / light_dist;
        let ndotl = (ndotl / light_dist).abs();
        let weight = Brdf1::eval() * ndotl * light.intensity / light_dist_sq;

        if weight != Vec3::zero() {
            Some((dir_in, light_dist, weight))
        } else {
            None
        }
    }

    #[must_use]
    pub fn sample_diffuse_ray<R: Rng>(dir_out: &Vec3, normal: &Vec3, rng: &mut R) -> (Vec3, Vec3) {
        let world_to_local = if *normal == -Vec3::unit_z() {
            Rotor3::from_rotation_xz(PI)
        } else {
            Rotor3::from_rotation_between(*normal, Vec3::unit_z())
        };
        let local_to_world = world_to_local.reversed();
        let dir_out_local = world_to_local * *dir_out;

        let (dir_in_local, pdf, brdf) = Brdf1::sample_eval(&dir_out_local, rng);
        let dir_in = local_to_world * dir_in_local;
        let ndotl = normal.dot(dir_in).abs();
        let weight = brdf * ndotl / pdf;

        debug_assert!(dir_in.as_array().iter().all(|f| f.is_normal()));
        (dir_in, weight)
    }
}

macro_rules! integrators_common_n {
    ($(($n:ident, $t:ident, $vt:ident, $rt:ident, $bt:ident)),+) => {
        $(
            pub struct $n;

            impl $n {
                #[must_use]
                pub fn sample_light(
                    dir_out: &$vt,
                    normal: &$vt,
                    hit_pos: &$vt,
                    valid: &$t,
                    light: &PointLight,
                ) -> ($vt, $t, $vt, $t) {
                    let light_vec = $vt::splat(light.pos) - *hit_pos;
                    let ndotl = normal.dot(light_vec);
                    let mut mask = (ndotl * normal.dot(*dir_out)).cmp_gt($t::ZERO) & valid;
                    if mask.none() {
                        return ($vt::zero(), $t::ZERO, $vt::zero(), mask);
                    }

                    let light_dist_sq = light_vec.mag_sq();
                    let light_dist = light_dist_sq.sqrt();
                    let dir_in = light_vec / light_dist;
                    let ndotl = (ndotl / light_dist).abs();
                    let weight = $bt::eval() * ndotl * $vt::splat(light.intensity) / light_dist_sq;

                    mask &= weight.mag_sq().cmp_gt($t::ZERO);

                    (dir_in, light_dist, weight, mask)
                }

                #[must_use]
                pub fn sample_diffuse_ray<R: Rng>(
                    dir_out: &$vt,
                    normal: &$vt,
                    rng: &mut R,
                ) -> ($vt, $vt) {
                    let neg_z = -$vt::unit_z();
                    let neg_z_mask = normal.x.cmp_eq(neg_z.x) & normal.y.cmp_eq(neg_z.y) & normal.z.cmp_eq(neg_z.z);
                    let world_to_local = neg_z_mask.blend_rotor(
                        &$rt::from_rotation_xz($t::PI),
                        &$rt::from_rotation_between(*normal, $vt::unit_z()),
                    );
                    let local_to_world = world_to_local.reversed();

                    let dir_out_local = world_to_local * *dir_out;
                    let (dir_in_local, pdf, brdf) = $bt::sample_eval(&dir_out_local, rng);
                    let dir_in = local_to_world * dir_in_local;
                    let ndotl = normal.dot(dir_in).abs();
                    let weight = brdf * ndotl / pdf;

                    (dir_in, weight)
                }
            }
        )+
    }
}

integrators_common_n!(
    (IntegratorsCommon4, f32x4, Vec3x4, Rotor3x4, Brdf4),
    (IntegratorsCommon8, f32x8, Vec3x8, Rotor3x8, Brdf8)
);
