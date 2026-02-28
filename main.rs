mod aabb;
mod brdf;
mod bvh;
mod camera;
mod light;
mod ray;
mod trace_stats;
mod triangle;
mod triangle_opt;

use crate::brdf::Brdf;
use crate::bvh::Bvh;
use crate::camera::{Camera, Rect};
use crate::light::PointLight;
use crate::ray::Ray;
use crate::trace_stats::TraceStats;
use crate::triangle::Triangle;
use chrono::Local;
use clap::Parser;
use core::f32;
use image::RgbImage;
use obj::Obj;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::fs;
use std::time::Instant;
use ultraviolet::{Rotor3, Vec3, Vec3x4, Vec4};
use wide::{CmpGe, f32x4};

const BATCH_SIZE: u32 = 64;
const RNG_SEED: u64 = 1235468;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short = 's', long, default_value_t = 128)]
    samples: u32,

    #[arg(short = 'b', long, default_value_t = 1)]
    max_bounces: u8,

    #[arg(long, default_value_t = false)]
    singlethread: bool,
}

pub struct PathInfo {
    weight: Vec3,
    destination_idx: u32,
    bounces: u8,
}

fn linear_to_gamma(v: Vec3) -> Vec3 {
    v.map(|f| f.sqrt())
}

fn color_vec_to_rgb_norm_gamma(v: Vec4) -> image::Rgb<u8> {
    let v = linear_to_gamma(v.xyz() / v.w) * 255.;
    image::Rgb([v.x as u8, v.y as u8, v.z as u8])
}

fn trace_batch<R: Rng>(
    batch: &mut Batch<R>,
    bvh: &Bvh,
    objects: &[Triangle],
    camera: &Camera,
    light: &PointLight,
    samples: u32,
    max_bounces: u8,
) {
    const SUBSAMPLES_PER_ITER: usize = 4;

    let num_pixels = (batch.region.width * batch.region.height) as usize;
    let max_rays = num_pixels * SUBSAMPLES_PER_ITER;
    let mut rays = Vec::with_capacity(max_rays);
    let mut ray_infos = Vec::with_capacity(max_rays);

    let max_shadow_rays = num_pixels * SUBSAMPLES_PER_ITER;
    let mut shadow_rays = Vec::with_capacity(max_shadow_rays);
    let mut shadow_ray_infos = Vec::with_capacity(max_shadow_rays);
    let mut hit_objects = Vec::with_capacity(max_rays.max(max_shadow_rays));
    for _ in 0..samples.div_ceil(SUBSAMPLES_PER_ITER as u32) {
        camera.generate_rays_4sp(&batch.region, &mut batch.rng, &mut rays, &mut ray_infos);
        for p in &mut batch.pixels {
            p.w += SUBSAMPLES_PER_ITER as f32;
        }

        while !rays.is_empty() {
            hit_objects.clear();
            hit_objects.resize(rays.len(), None);

            bvh.trace_stream(&mut rays, &mut hit_objects, &mut batch.trace_stats);

            shadow_rays.clear();
            shadow_ray_infos.clear();

            let mut new_rays_len = 0;
            for (i, hit_obj_idx) in hit_objects.iter().enumerate() {
                if let &Some(hit_obj_idx) = hit_obj_idx {
                    let ray = &rays[i];
                    let path_info = &ray_infos[i];
                    let hit_obj = &objects[hit_obj_idx as usize];
                    let hit_pos = ray.hit_pos();

                    // Shadow ray
                    let shadow_ray_dir = (light.pos - hit_pos).normalized();
                    let shadow_weight = Brdf::eval() * hit_obj.normal.dot(shadow_ray_dir).max(0.);
                    if shadow_weight != Vec3::zero() {
                        shadow_rays.push(Ray::new(&hit_pos, &shadow_ray_dir, 0.));
                        shadow_ray_infos.push(PathInfo {
                            weight: path_info.weight * shadow_weight,
                            ..*path_info
                        });
                    }

                    // Diffuse ray
                    if path_info.bounces < max_bounces {
                        let (dir_in_tangent, pdf, brdf) = Brdf::sample_eval(&mut batch.rng);
                        let tangent_to_world =
                            Rotor3::from_rotation_between(Vec3::unit_z(), hit_obj.normal);
                        let dir_in = tangent_to_world * dir_in_tangent;
                        let ndotl = hit_obj.normal.dot(dir_in);
                        let weight = brdf * ndotl / pdf;
                        rays[new_rays_len] = Ray::new(&hit_pos, &dir_in, 0.);
                        ray_infos[new_rays_len] = PathInfo {
                            weight: path_info.weight * weight,
                            destination_idx: path_info.destination_idx,
                            bounces: path_info.bounces + 1,
                        };
                        new_rays_len += 1;
                    }
                }
            }
            rays.truncate(new_rays_len);
            ray_infos.truncate(new_rays_len);

            if !shadow_rays.is_empty() {
                hit_objects.clear();
                hit_objects.resize(shadow_rays.len(), None);
                bvh.trace_stream(&mut shadow_rays, &mut hit_objects, &mut batch.trace_stats);

                for (ray_info, ray) in shadow_ray_infos.iter().zip(shadow_rays.iter()) {
                    if ray.hit_dist() == f32::INFINITY {
                        *batch.pixels[ray_info.destination_idx as usize] += Vec4::from(
                            ray_info.weight * light.intensity
                                / (light.pos - ray.origin_near.xyz()).mag_sq(),
                        );
                    }
                }
            }
        }
    }
}

fn load_scene() -> Vec<Triangle> {
    let load_start = Instant::now();
    let obj = Obj::load("./scenes/asian_dragon_obj/asian_dragon.obj").unwrap();

    let mut triangles = vec![];
    for o in &obj.data.objects {
        for g in &o.groups {
            triangles.extend(g.polys.iter().map(|p| {
                if p.0.len() != 3 {
                    panic!();
                }

                let verts = [
                    Vec3::from(obj.data.position[p.0[0].0]),
                    Vec3::from(obj.data.position[p.0[1].0]),
                    Vec3::from(obj.data.position[p.0[2].0]),
                ];

                Triangle::new(verts)
            }));
        }
    }
    let elapsed = load_start.elapsed();
    println!(
        "model load {} triangles in {:.2?}",
        triangles.len(),
        elapsed
    );

    triangles
}

struct Batch<'a, R: Rng> {
    region: Rect,
    pixels: Vec<&'a mut Vec4>,
    rng: R,
    trace_stats: TraceStats,
}

impl<R: Rng> Batch<'_, R> {
    fn new(region: Rect, rng: R) -> Self {
        Batch {
            region,
            pixels: Vec::new(),
            rng,
            trace_stats: TraceStats::new(),
        }
    }
}

fn generate_batches<'a, R: Rng>(
    pixels: &'a mut [Vec4],
    image_width: u32,
    image_height: u32,
    batch_size: u32,
    rng: &mut R,
) -> Vec<Batch<'a, SmallRng>> {
    let num_regions_x = image_width.div_ceil(batch_size);
    let num_regions_y = image_height.div_ceil(batch_size);

    let mut batches = Vec::with_capacity((num_regions_x * num_regions_y) as usize);
    for region_y in 0..num_regions_y {
        for region_x in 0..num_regions_x {
            let x = region_x * batch_size;
            let y = region_y * batch_size;
            let width = batch_size.min(image_width - x);
            let height = batch_size.min(image_height - y);
            let rect = Rect {
                x,
                y,
                width,
                height,
            };
            batches.push(Batch::new(rect, SmallRng::from_rng(rng)));
        }
    }

    for (i, pixel) in pixels.iter_mut().enumerate() {
        let x = (i as u32) % image_width;
        let y = (i as u32) / image_width;
        let region_x = x / batch_size;
        let region_y = y / batch_size;
        let region_idx = (region_y * num_regions_x + region_x) as usize;
        batches[region_idx].pixels.push(pixel);
    }

    batches
}

fn main() {
    let args = Args::parse();

    let triangles = load_scene();
    let bvh_start = Instant::now();
    let bvh = Bvh::build(&triangles);
    println!("bvh build {:.2?}", bvh_start.elapsed());

    let cam_pos = Vec3::new(0.6, 0.25, -1.).normalized() * 2500.;
    let cam_target = Vec3::new(0., 350., 0.);
    let light = PointLight::new(Vec3::new(5000., 5000., -10000.), Vec3::one(), 3e8);
    let image_width = 1920;
    let image_height = 1080;

    let camera = Camera::new(
        cam_pos,
        cam_target,
        Vec3::new(0., 1., 0.),
        60.,
        image_width,
        image_height,
    );

    let mut rng = SmallRng::seed_from_u64(RNG_SEED);
    let mut pixels = vec![Vec4::zero(); (image_width * image_height) as usize];
    let mut batches =
        generate_batches(&mut pixels, image_width, image_height, BATCH_SIZE, &mut rng);

    println!(
        "{} {}x{} batches, {} pixels/batch",
        batches.len(),
        BATCH_SIZE,
        BATCH_SIZE,
        BATCH_SIZE * BATCH_SIZE,
    );

    let time_start = Instant::now();
    let trace_fn = |batch| {
        trace_batch(
            batch,
            &bvh,
            &triangles,
            &camera,
            &light,
            args.samples,
            args.max_bounces,
        );
    };

    if args.singlethread {
        batches.iter_mut().for_each(trace_fn);
    } else {
        batches.par_iter_mut().for_each(trace_fn);
    }

    let elapsed = time_start.elapsed();
    println!(
        "{:.2?} for {} samples, {:.2} samples/sec",
        elapsed,
        args.samples,
        args.samples as f32 / elapsed.as_secs_f32()
    );

    let trace_stats = batches
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
