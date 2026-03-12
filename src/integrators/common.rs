use std::f32::consts::PI;

use rand::Rng;
use ultraviolet::{Rotor3, Rotor3x4, Vec3, Vec3x4};
use wide::{CmpEq as _, CmpGt as _, f32x4};

use crate::{brdf::Brdf, light::PointLight, utils::rotor_blend};

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
    let weight = Brdf::eval() * ndotl * light.intensity / light_dist_sq;

    if weight != Vec3::zero() {
        Some((dir_in, light_dist, weight))
    } else {
        None
    }
}

#[must_use]
pub fn sample_light4(
    dir_out: &Vec3x4,
    normal: &Vec3x4,
    hit_pos: &Vec3x4,
    light: &PointLight,
) -> (Vec3x4, f32x4, Vec3x4, u32) {
    let light_vec = Vec3x4::splat(light.pos) - *hit_pos;
    let ndotl = normal.dot(light_vec);
    let mut mask = ((ndotl * normal.dot(*dir_out)).cmp_gt(f32x4::ZERO)).move_mask() as u32;
    if mask == 0 {
        return (Vec3x4::zero(), f32x4::ZERO, Vec3x4::zero(), mask);
    }

    let light_dist_sq = light_vec.mag_sq();
    let light_dist = light_dist_sq.sqrt();
    let dir_in = light_vec / light_dist;
    let ndotl = (ndotl / light_dist).abs();
    let weight =
        Vec3x4::splat(Brdf::eval()) * ndotl * Vec3x4::splat(light.intensity) / light_dist_sq;

    mask &= weight.mag_sq().cmp_gt(f32x4::ZERO).move_mask() as u32;

    (dir_in, light_dist, weight, mask)
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

    let (dir_in_local, pdf, brdf) = Brdf::sample_eval(&dir_out_local, rng);
    let dir_in = local_to_world * dir_in_local;
    let ndotl = normal.dot(dir_in).abs();
    let weight = brdf * ndotl / pdf;

    debug_assert!(dir_in.as_array().iter().all(|f| f.is_normal()));
    (dir_in, weight)
}

// TODO: broken
#[must_use]
pub fn sample_diffuse_ray4<R: Rng>(
    dir_out: &Vec3x4,
    normal: &Vec3x4,
    ray_valid: u32,
    rng: &mut R,
) -> (Vec3x4, Vec3x4) {
    let neg_z = -Vec3x4::unit_z();
    let neg_z_mask = normal.x.cmp_eq(neg_z.x) & normal.y.cmp_eq(neg_z.y) & normal.z.cmp_eq(neg_z.z);
    let world_to_local = rotor_blend(
        Rotor3x4::from_rotation_xz(f32x4::PI),
        Rotor3x4::from_rotation_between(*normal, Vec3x4::unit_z()),
        neg_z_mask,
    );
    let local_to_world = world_to_local.reversed();

    let dir_out_local: [Vec3; 4] = (world_to_local * *dir_out).into();
    let mut dir_in_local = [Vec3::zero(); 4];
    let mut pdf = [0.; 4];
    let mut brdf = [Vec3::zero(); 4];
    for (i, d) in dir_out_local.into_iter().enumerate() {
        if ray_valid & 1 << i != 0 {
            let (d, p, b) = Brdf::sample_eval(&d, rng);
            dir_in_local[i] = d;
            pdf[i] = p;
            brdf[i] = b;
        }
    }
    let dir_in = local_to_world * Vec3x4::from(dir_in_local);
    let ndotl = normal.dot(dir_in).abs();
    let weight = Vec3x4::from(brdf) * ndotl / f32x4::from(pdf);

    (dir_in, weight)
}
