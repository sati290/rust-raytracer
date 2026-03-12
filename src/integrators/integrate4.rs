use core::f32;

use rand::Rng;
use ultraviolet::{Vec3, Vec3x4, Vec4};
use wide::f32x4;

use crate::{
    Tile,
    camera::CameraRayGenerator,
    integrators::common::{SHADOW_RAY_NEAR, sample_diffuse_ray4, sample_light4},
    ray::Ray4,
    scene::Scene,
    trace_stats::TraceStats,
};

fn integrate_ray4<R: Rng>(
    camera_ray: Ray4,
    scene: &Scene,
    max_bounces: u8,
    dest_pixel: &mut Vec4,
    rng: &mut R,
    trace_stats: &mut TraceStats,
) {
    let Scene {
        mesh,
        bvh,
        camera: _,
        light,
    } = scene;

    let mut bounces = 0;
    let mut weight = Vec3x4::one();
    let mut ray = camera_ray;
    let mut ray_valid = 0b1111;

    loop {
        let hit = bvh.intersect4(&mut ray, ray_valid, trace_stats);
        let mut normal = [Vec3::zero(); 4];
        for (i, h) in hit.into_iter().enumerate() {
            if let Some(h) = h {
                let hit_obj = mesh.get_triangle(h);
                normal[i] = *hit_obj.normal();
            } else {
                ray_valid &= !(1 << i);
            }
        }

        if ray_valid == 0 {
            break;
        }

        let hit_pos = ray.hit_pos();
        let dir_out = -ray.direction;
        let normal = Vec3x4::from(normal);

        // Shadow ray
        let (shadow_ray_dir, shadow_far, shadow_weight, shadow_valid) =
            sample_light4(&dir_out, &normal, &hit_pos, light);
        let shadow_valid = shadow_valid & ray_valid;
        if shadow_valid != 0 {
            let shadow_ray = Ray4::new(
                &hit_pos,
                &shadow_ray_dir,
                &f32x4::splat(SHADOW_RAY_NEAR),
                &shadow_far,
            );
            let occluded = bvh.occluded4(&shadow_ray, shadow_valid, trace_stats);
            let light_valid = shadow_valid & !occluded;
            let light_weight: [Vec3; 4] = (weight * shadow_weight).into();
            for (i, w) in light_weight.into_iter().enumerate() {
                if light_valid & 1 << i != 0 {
                    *dest_pixel += Vec4::from(w);
                }
            }
        }

        // Diffuse ray
        if bounces >= max_bounces {
            break;
        }

        let (dir_in, diffuse_weight) = sample_diffuse_ray4(&dir_out, &normal, ray_valid, rng);
        ray = Ray4::new(
            &hit_pos,
            &dir_in,
            &f32x4::splat(SHADOW_RAY_NEAR),
            &f32x4::splat(f32::INFINITY),
        );
        bounces += 1;
        weight *= diffuse_weight;
    }
}

pub fn integrate_tile4<R: Rng>(
    tile: &mut Tile<R>,
    viewport_size: (u32, u32),
    scene: &Scene,
    samples: u32,
    max_bounces: u8,
) {
    for p in &mut tile.pixels {
        p.w += samples as f32;
    }

    let cam_pos4 = Vec3x4::splat(scene.camera.position());
    let mut ray_generator =
        CameraRayGenerator::new(&scene.camera, viewport_size.0, viewport_size.1, tile.region);

    while !ray_generator.is_done() {
        let dest_pixel = &mut tile.pixels[ray_generator.current_pixel_idx() as usize];
        for _ in 0..samples.div_ceil(4) {
            let ray_dir = ray_generator.sample4(&mut tile.rng);

            integrate_ray4(
                Ray4::new(
                    &cam_pos4,
                    &ray_dir,
                    &f32x4::ZERO,
                    &f32x4::splat(f32::INFINITY),
                ),
                scene,
                max_bounces,
                dest_pixel,
                &mut tile.rng,
                &mut tile.trace_stats,
            );
        }

        ray_generator.next_pixel();
    }
}
