use std::f32::consts::PI;

use rand::Rng;
use ultraviolet::{Rotor3, Vec3};

use crate::{
    brdf::Brdf,
    camera::{Camera, CameraRayGenerator, Rect},
    light::PointLight,
    ray::Ray,
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

pub fn generate_rays<R: Rng>(
    camera: &Camera,
    viewport_size: (u32, u32),
    region: &Rect,
    samples: u32,
    rng: &mut R,
    rays: &mut Vec<Ray>,
    path_infos: &mut Vec<PathInfo>,
) {
    let mut generator = CameraRayGenerator::new(camera, viewport_size.0, viewport_size.1, *region);
    while !generator.is_done() {
        for _ in 0..samples / 8 {
            let dirs: [Vec3; 8] = generator.sample8(rng).into();
            for d in dirs {
                rays.push(Ray::new(&camera.position(), &d, 0., f32::INFINITY));
                path_infos.push(PathInfo {
                    weight: Vec3::one(),
                    destination_idx: generator.current_pixel_idx(),
                    bounces: 0,
                });
            }
        }

        for _ in 0..samples % 8 {
            let dir = generator.sample(rng);
            rays.push(Ray::new(&camera.position(), &dir, 0., f32::INFINITY));
            path_infos.push(PathInfo {
                weight: Vec3::one(),
                destination_idx: generator.current_pixel_idx(),
                bounces: 0,
            });
        }

        generator.next_pixel();
    }
}

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

    Some((dir_in, light_dist, weight))
}

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
