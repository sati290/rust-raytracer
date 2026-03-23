use nalgebra::SimdValue;
use rand::Rng;
use simba::simd::WideF32x8;

use crate::{
    Tile,
    camera::{Camera, CameraRayGenerator, Rect},
    integrators::common::*,
    math::{Vec3f, Vec4f},
    ray::StreamRay,
    scene::Scene,
};

pub fn generate_rays<R: Rng>(
    camera: &Camera,
    viewport_size: (u32, u32),
    region: &Rect,
    samples: u32,
    rng: &mut R,
    rays: &mut Vec<StreamRay>,
    path_infos: &mut Vec<PathInfo>,
) {
    let mut generator = CameraRayGenerator::new(camera, viewport_size.0, viewport_size.1, *region);
    while !generator.is_done() {
        for _ in 0..samples / 8 {
            let dirs = generator.sample::<WideF32x8>(rng);
            for d in 0..8 {
                rays.push(StreamRay::new(
                    &camera.position(),
                    &dirs.extract(d),
                    0.,
                    f32::INFINITY,
                ));
                path_infos.push(PathInfo {
                    weight: Vec3f::from_element(1.),
                    destination_idx: generator.current_pixel_idx(),
                    bounces: 0,
                });
            }
        }

        for _ in 0..samples % 8 {
            let dir = generator.sample1(rng);
            rays.push(StreamRay::new(&camera.position(), &dir, 0., f32::INFINITY));
            path_infos.push(PathInfo {
                weight: Vec3f::from_element(1.),
                destination_idx: generator.current_pixel_idx(),
                bounces: 0,
            });
        }

        generator.next_pixel();
    }
}

pub fn integrate_tile_stream<R: Rng>(
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

    let max_rays = (tile.region.width * tile.region.height * samples) as usize;
    let mut rays = Vec::with_capacity(max_rays);
    let mut ray_infos = Vec::with_capacity(max_rays);
    let mut hit_objects = Vec::with_capacity(max_rays);

    generate_rays(
        camera,
        viewport_size,
        &tile.region,
        samples,
        &mut tile.rng,
        &mut rays,
        &mut ray_infos,
    );

    for p in &mut tile.pixels {
        p.w += samples as f32;
    }

    let max_shadow_rays = max_rays;
    let mut shadow_rays = Vec::with_capacity(max_shadow_rays);
    let mut shadow_ray_infos = Vec::with_capacity(max_shadow_rays);
    let mut shadow_rays_occluded = Vec::with_capacity(max_shadow_rays);

    while !rays.is_empty() {
        hit_objects.clear();
        hit_objects.resize(rays.len(), None);

        bvh.intersect_stream(&mut rays, &mut hit_objects, &mut tile.trace_stats);

        shadow_rays.clear();
        shadow_ray_infos.clear();

        let mut new_rays_len = 0;
        for (i, hit_obj_idx) in hit_objects.iter().enumerate() {
            if let &Some(hit_obj_idx) = hit_obj_idx {
                let ray = &rays[i];
                let path_info = &ray_infos[i];
                let hit_obj = mesh.get_triangle(hit_obj_idx);
                let hit_pos = ray.hit_pos();
                let dir_out = -ray.direction.xyz();
                let normal = hit_obj.normal();

                // Shadow ray
                if let Some((shadow_ray_dir, shadow_far, shadow_weight)) =
                    IntegratorsCommon1::sample_light(&dir_out, normal, &hit_pos, light)
                {
                    shadow_rays.push(StreamRay::new(
                        &hit_pos,
                        &shadow_ray_dir,
                        SHADOW_RAY_NEAR,
                        shadow_far,
                    ));
                    shadow_ray_infos.push(PathInfo {
                        weight: path_info.weight.component_mul(&shadow_weight),
                        ..*path_info
                    });
                }

                // Diffuse ray
                if path_info.bounces < max_bounces {
                    let (dir_in, weight) =
                        IntegratorsCommon1::sample_diffuse_ray(&dir_out, normal, &mut tile.rng);
                    rays[new_rays_len] =
                        StreamRay::new(&hit_pos, &dir_in, SHADOW_RAY_NEAR, f32::INFINITY);
                    ray_infos[new_rays_len] = path_info.diffuse(&weight);
                    new_rays_len += 1;
                }
            }
        }
        rays.truncate(new_rays_len);
        ray_infos.truncate(new_rays_len);

        if !shadow_rays.is_empty() {
            shadow_rays_occluded.clear();
            shadow_rays_occluded.resize(shadow_rays.len(), false);
            bvh.occluded_stream(
                &mut shadow_rays,
                &mut shadow_rays_occluded,
                &mut tile.trace_stats,
            );

            for (occluded, path_info) in shadow_rays_occluded.iter().zip(shadow_ray_infos.iter()) {
                if !occluded {
                    *tile.pixels[path_info.destination_idx as usize] += Vec4f::new(
                        path_info.weight.x,
                        path_info.weight.y,
                        path_info.weight.z,
                        0.,
                    );
                }
            }
        }
    }
}

pub fn integrate_tile_stream_shadow_immediate<R: Rng>(
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

    let max_rays = (tile.region.width * tile.region.height * samples) as usize;
    let mut rays = Vec::with_capacity(max_rays);
    let mut ray_infos = Vec::with_capacity(max_rays);
    let mut hit_objects = Vec::with_capacity(max_rays);

    generate_rays(
        camera,
        viewport_size,
        &tile.region,
        samples,
        &mut tile.rng,
        &mut rays,
        &mut ray_infos,
    );

    for p in &mut tile.pixels {
        p.w += samples as f32;
    }

    while !rays.is_empty() {
        hit_objects.clear();
        hit_objects.resize(rays.len(), None);

        bvh.intersect_stream(&mut rays, &mut hit_objects, &mut tile.trace_stats);

        let mut new_rays_len = 0;
        for (i, hit_obj_idx) in hit_objects.iter().enumerate() {
            if let &Some(hit_obj_idx) = hit_obj_idx {
                let ray = &rays[i];
                let path_info = &ray_infos[i];
                let hit_obj = mesh.get_triangle(hit_obj_idx);
                let hit_pos = ray.hit_pos();
                let dir_out = -ray.direction.xyz();
                let normal = hit_obj.normal();

                // Shadow ray
                if let Some((shadow_ray_dir, shadow_far, shadow_weight)) =
                    IntegratorsCommon1::sample_light(&dir_out, normal, &hit_pos, light)
                {
                    let shadow_ray =
                        StreamRay::new(&hit_pos, &shadow_ray_dir, SHADOW_RAY_NEAR, shadow_far);
                    if !bvh.occluded1(&shadow_ray, &mut tile.trace_stats) {
                        let contrib = path_info.weight.component_mul(&shadow_weight);
                        *tile.pixels[path_info.destination_idx as usize] +=
                            Vec4f::new(contrib.x, contrib.y, contrib.z, 0.);
                    }
                }

                // Diffuse ray
                if path_info.bounces < max_bounces {
                    let (dir_in, weight) =
                        IntegratorsCommon1::sample_diffuse_ray(&dir_out, normal, &mut tile.rng);
                    rays[new_rays_len] =
                        StreamRay::new(&hit_pos, &dir_in, SHADOW_RAY_NEAR, f32::INFINITY);
                    ray_infos[new_rays_len] = path_info.diffuse(&weight);
                    new_rays_len += 1;
                }
            }
        }
        rays.truncate(new_rays_len);
        ray_infos.truncate(new_rays_len);
    }
}

pub fn integrate_stream_camera_only<R: Rng>(
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

    let max_rays = (tile.region.width * tile.region.height * samples) as usize;
    let mut rays = Vec::with_capacity(max_rays);
    let mut ray_infos = Vec::with_capacity(max_rays);

    generate_rays(
        camera,
        viewport_size,
        &tile.region,
        samples,
        &mut tile.rng,
        &mut rays,
        &mut ray_infos,
    );

    for p in &mut tile.pixels {
        p.w += samples as f32;
    }

    let mut hit_objects = vec![None; rays.len()];
    bvh.intersect_stream(&mut rays, &mut hit_objects, &mut tile.trace_stats);

    for ((camera_hit_obj_idx, camera_ray), camera_path_info) in hit_objects
        .into_iter()
        .zip(rays.into_iter())
        .zip(ray_infos.into_iter())
    {
        let mut hit = camera_hit_obj_idx;
        let mut ray = camera_ray;
        let mut path_info = camera_path_info;

        while let Some(hit_obj_idx) = hit {
            let hit_obj = mesh.get_triangle(hit_obj_idx);
            let hit_pos = ray.hit_pos();
            let dir_out = -ray.direction.xyz();
            let normal = hit_obj.normal();

            // Shadow ray
            if let Some((shadow_ray_dir, shadow_far, shadow_weight)) =
                IntegratorsCommon1::sample_light(&dir_out, normal, &hit_pos, light)
            {
                let shadow_ray =
                    StreamRay::new(&hit_pos, &shadow_ray_dir, SHADOW_RAY_NEAR, shadow_far);
                if !bvh.occluded1(&shadow_ray, &mut tile.trace_stats) {
                    let contrib = path_info.weight.component_mul(&shadow_weight);
                    *tile.pixels[path_info.destination_idx as usize] +=
                        Vec4f::new(contrib.x, contrib.y, contrib.z, 0.);
                }
            }

            // Diffuse ray
            if path_info.bounces >= max_bounces {
                break;
            }

            let (dir_in, weight) =
                IntegratorsCommon1::sample_diffuse_ray(&dir_out, normal, &mut tile.rng);
            ray = StreamRay::new(&hit_pos, &dir_in, SHADOW_RAY_NEAR, f32::INFINITY);
            hit = scene.bvh.intersect1(&mut ray, &mut tile.trace_stats);
            path_info = path_info.diffuse(&weight);
        }
    }
}
