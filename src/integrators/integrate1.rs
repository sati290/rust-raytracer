use core::f32;

use rand::Rng;
use ultraviolet::{Vec3, Vec4};

use crate::{
    Tile,
    camera::CameraRayGenerator,
    integrators::common::{SHADOW_RAY_NEAR, sample_diffuse_ray, sample_light},
    ray::StreamRay,
    scene::Scene,
    trace_stats::TraceStats,
};

fn integrate_ray1<R: Rng>(
    camera_ray: StreamRay,
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
    let mut weight = Vec3::one();
    let mut ray = camera_ray;

    while let Some(hit) = bvh.intersect1(&mut ray, trace_stats) {
        let hit_obj = mesh.get_triangle(hit);
        let hit_pos = ray.hit_pos();
        let dir_out = -ray.direction.xyz();
        let normal = hit_obj.normal();

        // Shadow ray
        if let Some((shadow_ray_dir, shadow_far, shadow_weight)) =
            sample_light(&dir_out, normal, &hit_pos, light)
        {
            let shadow_ray = StreamRay::new(&hit_pos, &shadow_ray_dir, SHADOW_RAY_NEAR, shadow_far);
            if !bvh.occluded1(&shadow_ray, trace_stats) {
                *dest_pixel += Vec4::from(weight * shadow_weight);
            }
        }

        // Diffuse ray
        if bounces >= max_bounces {
            break;
        }

        let (dir_in, diffuse_weight) = sample_diffuse_ray(&dir_out, normal, rng);
        ray = StreamRay::new(&hit_pos, &dir_in, SHADOW_RAY_NEAR, f32::INFINITY);
        bounces += 1;
        weight *= diffuse_weight;
    }
}

pub fn integrate_tile1<R: Rng>(
    tile: &mut Tile<R>,
    viewport_size: (u32, u32),
    scene: &Scene,
    samples: u32,
    max_bounces: u8,
) {
    for p in &mut tile.pixels {
        p.w += samples as f32;
    }

    let mut ray_generator =
        CameraRayGenerator::new(&scene.camera, viewport_size.0, viewport_size.1, tile.region);

    while !ray_generator.is_done() {
        let dest_pixel = &mut tile.pixels[ray_generator.current_pixel_idx() as usize];
        for _ in 0..samples {
            let ray_dir = ray_generator.sample(&mut tile.rng);

            integrate_ray1(
                StreamRay::new(&scene.camera.position(), &ray_dir, 0., f32::INFINITY),
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
