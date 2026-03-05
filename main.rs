mod aabb;
mod args;
mod brdf;
mod bvh;
mod camera;
mod light;
mod ray;
mod scene;
mod trace_stats;
mod triangle;
mod triangle_opt;

use crate::args::{Args, TraceMode};
use crate::brdf::Brdf;
use crate::camera::{Camera, CameraRayGenerator, Rect};
use crate::light::PointLight;
use crate::ray::Ray;
use crate::scene::{SCENE_ASIAN_DRAGON, SCENE_SANMIGUEL, Scene};
use crate::trace_stats::TraceStats;
use chrono::Local;
use clap::Parser as _;
use core::f32;
use image::RgbImage;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::f32::consts::PI;
use std::fs;
use std::time::Instant;
use ultraviolet::{Rotor3, Vec3, Vec3x4, Vec4};
use wide::{CmpGe, f32x4};

const SHADOW_RAY_NEAR: f32 = 1e-5;

#[derive(Clone)]
pub struct PathInfo {
    weight: Vec3,
    destination_idx: u32,
    bounces: u8,
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

fn linear_to_gamma(v: Vec3) -> Vec3 {
    v.map(|f| f.sqrt())
}

fn color_vec_to_rgb_norm_gamma(v: Vec4) -> image::Rgb<u8> {
    let v = linear_to_gamma(v.xyz() / v.w) * 255.;
    image::Rgb([v.x as u8, v.y as u8, v.z as u8])
}

fn generate_rays<R: Rng>(
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

fn sample_light(
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

fn sample_diffuse_ray<R: Rng>(dir_out: &Vec3, normal: &Vec3, rng: &mut R) -> (Vec3, Vec3) {
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

fn trace_tile_stream<R: Rng>(
    tile: &mut Tile<R>,
    viewport_size: (u32, u32),
    scene: &Scene,
    samples: u32,
    max_bounces: u8,
) {
    let Scene {
        objects,
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
                let hit_obj = &objects[hit_obj_idx as usize];
                let hit_pos = ray.hit_pos();
                let dir_out = -ray.direction.xyz();

                // Shadow ray
                if let Some((shadow_ray_dir, shadow_far, shadow_weight)) =
                    sample_light(&dir_out, &hit_obj.normal, &hit_pos, light)
                    && shadow_weight != Vec3::zero()
                {
                    shadow_rays.push(Ray::new(
                        &hit_pos,
                        &shadow_ray_dir,
                        SHADOW_RAY_NEAR,
                        shadow_far,
                    ));
                    shadow_ray_infos.push(PathInfo {
                        weight: path_info.weight * shadow_weight,
                        ..*path_info
                    });
                }

                // Diffuse ray
                if path_info.bounces < max_bounces {
                    let (dir_in, weight) =
                        sample_diffuse_ray(&dir_out, &hit_obj.normal, &mut tile.rng);
                    rays[new_rays_len] =
                        Ray::new(&hit_pos, &dir_in, SHADOW_RAY_NEAR, f32::INFINITY);
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
                    *tile.pixels[path_info.destination_idx as usize] +=
                        Vec4::from(path_info.weight);
                }
            }
        }
    }
}

fn trace_tile_immediate_shadow_rays<R: Rng>(
    tile: &mut Tile<R>,
    viewport_size: (u32, u32),
    scene: &Scene,
    samples: u32,
    max_bounces: u8,
) {
    let Scene {
        objects,
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
                let hit_obj = &objects[hit_obj_idx as usize];
                let hit_pos = ray.hit_pos();
                let dir_out = -ray.direction.xyz();

                // Shadow ray
                if let Some((shadow_ray_dir, shadow_far, shadow_weight)) =
                    sample_light(&dir_out, &hit_obj.normal, &hit_pos, light)
                    && shadow_weight != Vec3::zero()
                {
                    let shadow_ray =
                        Ray::new(&hit_pos, &shadow_ray_dir, SHADOW_RAY_NEAR, shadow_far);
                    if !bvh.occluded1(&shadow_ray, &mut tile.trace_stats) {
                        *tile.pixels[path_info.destination_idx as usize] +=
                            Vec4::from(path_info.weight * shadow_weight);
                    }
                }

                // Diffuse ray
                if path_info.bounces < max_bounces {
                    let (dir_in, weight) =
                        sample_diffuse_ray(&dir_out, &hit_obj.normal, &mut tile.rng);
                    rays[new_rays_len] =
                        Ray::new(&hit_pos, &dir_in, SHADOW_RAY_NEAR, f32::INFINITY);
                    ray_infos[new_rays_len] = path_info.diffuse(&weight);
                    new_rays_len += 1;
                }
            }
        }
        rays.truncate(new_rays_len);
        ray_infos.truncate(new_rays_len);
    }
}

fn trace_stream_camera_only<R: Rng>(
    tile: &mut Tile<R>,
    viewport_size: (u32, u32),
    scene: &Scene,
    samples: u32,
    max_bounces: u8,
) {
    let Scene {
        objects,
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
            let hit_obj = &objects[hit_obj_idx as usize];
            let hit_pos = ray.hit_pos();
            let dir_out = -ray.direction.xyz();

            // Shadow ray
            if let Some((shadow_ray_dir, shadow_far, shadow_weight)) =
                sample_light(&dir_out, &hit_obj.normal, &hit_pos, light)
                && shadow_weight != Vec3::zero()
            {
                let shadow_ray = Ray::new(&hit_pos, &shadow_ray_dir, SHADOW_RAY_NEAR, shadow_far);
                if !bvh.occluded1(&shadow_ray, &mut tile.trace_stats) {
                    *tile.pixels[path_info.destination_idx as usize] +=
                        Vec4::from(path_info.weight * shadow_weight);
                }
            }

            // Diffuse ray
            if path_info.bounces >= max_bounces {
                break;
            }

            let (dir_in, weight) = sample_diffuse_ray(&dir_out, &hit_obj.normal, &mut tile.rng);
            ray = Ray::new(&hit_pos, &dir_in, SHADOW_RAY_NEAR, f32::INFINITY);
            hit = scene.bvh.intersect1(&mut ray, &mut tile.trace_stats);
            path_info = path_info.diffuse(&weight);
        }
    }
}

struct Tile<'a, R: Rng> {
    region: Rect,
    pixels: Vec<&'a mut Vec4>,
    rng: R,
    trace_stats: TraceStats,
}

impl<R: Rng> Tile<'_, R> {
    fn new(region: Rect, rng: R) -> Self {
        Tile {
            region,
            pixels: Vec::new(),
            rng,
            trace_stats: TraceStats::new(),
        }
    }
}

fn generate_tiles<'a, R: Rng>(
    pixels: &'a mut [Vec4],
    image_width: u32,
    image_height: u32,
    tile_size: u32,
    rng: &mut R,
) -> Vec<Tile<'a, SmallRng>> {
    let num_regions_x = image_width.div_ceil(tile_size);
    let num_regions_y = image_height.div_ceil(tile_size);

    let mut batches = Vec::with_capacity((num_regions_x * num_regions_y) as usize);
    for region_y in 0..num_regions_y {
        for region_x in 0..num_regions_x {
            let x = region_x * tile_size;
            let y = region_y * tile_size;
            let width = tile_size.min(image_width - x);
            let height = tile_size.min(image_height - y);
            let rect = Rect {
                x,
                y,
                width,
                height,
            };
            batches.push(Tile::new(rect, SmallRng::from_rng(rng)));
        }
    }

    for (i, pixel) in pixels.iter_mut().enumerate() {
        let x = (i as u32) % image_width;
        let y = (i as u32) / image_width;
        let region_x = x / tile_size;
        let region_y = y / tile_size;
        let region_idx = (region_y * num_regions_x + region_x) as usize;
        batches[region_idx].pixels.push(pixel);
    }

    batches
}

fn main() {
    let args = Args::parse();

    let image_width = 1920;
    let image_height = 1080;
    let scene = Scene::load(match args.scene {
        args::Scene::AsianDragon => &SCENE_ASIAN_DRAGON,
        args::Scene::SanMiguel => &SCENE_SANMIGUEL,
    });

    let mut rng = SmallRng::seed_from_u64(args.seed);
    let mut pixels = vec![Vec4::zero(); (image_width * image_height) as usize];
    let mut tiles = generate_tiles(
        &mut pixels,
        image_width,
        image_height,
        args.tile_size,
        &mut rng,
    );

    println!("mode: {:?}", args.mode);
    println!(
        "{} {}x{} tiles, {} pixels/tile, {} samples/pixel, {} total samples/tile, max bounces {}",
        tiles.len(),
        args.tile_size,
        args.tile_size,
        args.tile_size * args.tile_size,
        args.samples,
        args.tile_size * args.tile_size * args.samples,
        args.max_bounces
    );

    let time_start = Instant::now();
    let trace_fn = |tile| {
        let trace_fn = match args.mode {
            TraceMode::Stream => trace_tile_stream,
            TraceMode::StreamShadowImmediate => trace_tile_immediate_shadow_rays,
            TraceMode::StreamCameraOnly => trace_stream_camera_only,
        };

        trace_fn(
            tile,
            (image_width, image_height),
            &scene,
            args.samples,
            args.max_bounces,
        );
    };

    if args.singlethread {
        tiles.iter_mut().for_each(trace_fn);
    } else {
        tiles.par_iter_mut().for_each(trace_fn);
    }

    let elapsed = time_start.elapsed();
    println!(
        "{:.2?} for {} samples, {:.2} samples/sec",
        elapsed,
        args.samples,
        args.samples as f32 / elapsed.as_secs_f32()
    );

    let trace_stats = tiles
        .iter()
        .fold(TraceStats::new(), |acc, x| acc + x.trace_stats);
    trace_stats.print();

    let image = RgbImage::from_fn(image_width, image_height, |x, y| {
        color_vec_to_rgb_norm_gamma(pixels[(y * image_width + x) as usize])
    });
    let now = Local::now();
    let _ = fs::create_dir("output");
    image
        .save(format!("output/{}.png", now.format("%Y%m%d_%H%M%S")))
        .unwrap();
}
