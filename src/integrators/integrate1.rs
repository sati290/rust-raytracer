use core::f32;

use rand::Rng;
use ultraviolet::{Vec3, Vec4};

use crate::{
    Tile,
    camera::CameraRayGenerator,
    integrators::common::{SHADOW_RAY_NEAR, sample_diffuse_ray, sample_light},
    ray::Ray,
    scene::Scene,
};

pub fn integrate_tile1<R: Rng>(
    tile: &mut Tile<R>,
    viewport_size: (u32, u32),
    scene: &Scene,
    samples: u32,
    max_bounces: u8,
) {
    let Scene {
        mesh,
        bvh,
        camera,
        light,
    } = scene;

    for p in &mut tile.pixels {
        p.w += samples as f32;
    }

    let mut ray_generator =
        CameraRayGenerator::new(camera, viewport_size.0, viewport_size.1, tile.region);

    while !ray_generator.is_done() {
        for _ in 0..samples {
            let ray_dir = ray_generator.sample(&mut tile.rng);
            let dest_idx = ray_generator.current_pixel_idx();
            let mut bounces = 0;
            let mut weight = Vec3::one();
            let mut ray = Ray::new(&camera.position(), &ray_dir, 0., f32::INFINITY);

            while let Some(hit) = bvh.intersect1(&mut ray, &mut tile.trace_stats) {
                let hit_obj = mesh.get_triangle(hit);
                let hit_pos = ray.hit_pos();
                let dir_out = -ray.direction.xyz();
                let normal = hit_obj.normal();

                // Shadow ray
                if let Some((shadow_ray_dir, shadow_far, shadow_weight)) =
                    sample_light(&dir_out, normal, &hit_pos, light)
                    && shadow_weight != Vec3::zero()
                {
                    let shadow_ray =
                        Ray::new(&hit_pos, &shadow_ray_dir, SHADOW_RAY_NEAR, shadow_far);
                    if !bvh.occluded1(&shadow_ray, &mut tile.trace_stats) {
                        *tile.pixels[dest_idx as usize] += Vec4::from(weight * shadow_weight);
                    }
                }

                // Diffuse ray
                if bounces >= max_bounces {
                    break;
                }

                let (dir_in, diffuse_weight) = sample_diffuse_ray(&dir_out, normal, &mut tile.rng);
                ray = Ray::new(&hit_pos, &dir_in, SHADOW_RAY_NEAR, f32::INFINITY);
                bounces += 1;
                weight *= diffuse_weight;
            }
        }

        ray_generator.next_pixel();
    }
}
