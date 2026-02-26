mod aabb;
mod bvh;
mod camera;
mod ray;
mod trace_stats;
mod triangle;
mod triangle_opt;

use crate::bvh::Bvh;
use crate::camera::{Camera, Rect};
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
use ultraviolet::{Vec3, Vec3x4, Vec4};
use wide::{CmpGe, f32x4};

const NUM_SUBSAMPLES: usize = 4;
const BATCH_SIZE: u32 = 64;
const RNG_SEED: u64 = 1235468;
const MAX_BOUNCES: u8 = 1;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    bench: bool,

    #[arg(short, long)]
    singlethread: bool,
}

pub struct PathInfo {
    contribution: Vec3,
    destination_idx: u32,
    bounces: u8,
}

fn color_vec_to_rgb_norm(v: Vec4) -> image::Rgb<u8> {
    let v = v.xyz() / v.w * 255.;
    image::Rgb([v.x as u8, v.y as u8, v.z as u8])
}

fn brdf(dir_in: Vec3, dir_out: Vec3, normal: Vec3) -> Vec3 {
    let ndotl = normal.dot(dir_in);
    let r = dir_in.reflected(normal);
    let s = r.dot(dir_out).powf(20.);

    Vec3::one() * (ndotl.clamp(0., 1.) + 0.2 * s.clamp(0., 1.))
}

fn _brdf_simd(dir_in: Vec3x4, _dir_out: Vec3x4, normal: Vec3x4) -> Vec3x4 {
    let ndotl = normal.dot(dir_in);
    Vec3x4::one() * ndotl.max(f32x4::ZERO)
}

fn trace_batch<R: Rng>(
    batch: &mut Batch<R>,
    bvh: &Bvh,
    objects: &[Triangle],
    camera: &Camera,
    light_pos: &Vec3,
) {
    let num_pixels = (batch.region.width * batch.region.height) as usize;
    let max_rays = num_pixels * NUM_SUBSAMPLES;
    let mut rays = Vec::with_capacity(max_rays);
    let mut ray_infos = Vec::with_capacity(max_rays);

    let max_shadow_rays = num_pixels * NUM_SUBSAMPLES * MAX_BOUNCES as usize;
    let mut shadow_rays = Vec::with_capacity(max_shadow_rays);
    let mut shadow_ray_infos = Vec::with_capacity(max_shadow_rays);
    let mut hit_objects = Vec::with_capacity(max_rays.max(max_shadow_rays));

    camera.generate_rays_4sp(&batch.region, &mut batch.rng, &mut rays, &mut ray_infos);
    for p in &mut batch.pixels {
        p.w += NUM_SUBSAMPLES as f32;
    }

    while !rays.is_empty() {
        hit_objects.clear();
        hit_objects.resize(rays.len(), None);

        bvh.trace_stream(&mut rays, &mut hit_objects, &mut batch.trace_stats);

        let mut new_rays_len = 0;
        for (i, hit_obj_idx) in hit_objects.iter().enumerate() {
            if let &Some(hit_obj_idx) = hit_obj_idx {
                let ray = &rays[i];
                let path_info = &ray_infos[i];
                let hit_obj = &objects[hit_obj_idx as usize];
                let hit_pos = ray.hit_pos();
                let ray_dir_inv = -ray.direction.xyz();

                // Shadow ray
                let shadow_ray_dir = (*light_pos - hit_pos).normalized();
                let shadow_brdf = brdf(shadow_ray_dir, ray_dir_inv, hit_obj.normal);
                shadow_rays.push(Ray::new(&hit_pos, &shadow_ray_dir, 0.));
                shadow_ray_infos.push(PathInfo {
                    contribution: path_info.contribution * shadow_brdf,
                    ..*path_info
                });

                // Diffuse ray
                if path_info.bounces < MAX_BOUNCES - 1 {
                    let diffuse_ray_dir = ray_dir_inv.reflected(hit_obj.normal);
                    let diffuse_brdf = brdf(diffuse_ray_dir, ray_dir_inv, hit_obj.normal);
                    rays[new_rays_len] = Ray::new(&hit_pos, &diffuse_ray_dir, 0.);
                    ray_infos[new_rays_len] = PathInfo {
                        contribution: path_info.contribution * diffuse_brdf,
                        destination_idx: path_info.destination_idx,
                        bounces: path_info.bounces + 1,
                    };
                    new_rays_len += 1;
                }
            }
        }
        rays.truncate(new_rays_len);
        ray_infos.truncate(new_rays_len);
    }

    hit_objects.clear();
    hit_objects.resize(shadow_rays.len(), None);
    bvh.trace_stream(&mut shadow_rays, &mut hit_objects, &mut batch.trace_stats);

    for (ray_info, ray) in shadow_ray_infos.into_iter().zip(shadow_rays.into_iter()) {
        if ray.hit_dist() == f32::INFINITY {
            *batch.pixels[ray_info.destination_idx as usize] += Vec4::from(ray_info.contribution);
        }
    }
}

fn load_scene() -> Vec<Triangle> {
    let load_start = Instant::now();

    // Model from http://graphics.stanford.edu/data/3Dscanrep/
    let obj = Obj::load("./data/asian_dragon_obj/asian_dragon.obj").unwrap();

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
    let light_pos = Vec3::new(5000., 5000., -10000.);
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

    const RAYS_PER_BATCH: usize = BATCH_SIZE as usize * BATCH_SIZE as usize * NUM_SUBSAMPLES;
    println!(
        "{} {}x{} batches, {} rays/batch",
        batches.len(),
        BATCH_SIZE,
        BATCH_SIZE,
        RAYS_PER_BATCH,
    );

    println!(
        "[Vec3; {0}x{0}] = {1} kbytes",
        BATCH_SIZE,
        std::mem::size_of::<[Vec3; (BATCH_SIZE * BATCH_SIZE) as usize]>() as f32 / 1024.
    );
    println!(
        "[Ray; {}] = {} kbytes",
        RAYS_PER_BATCH,
        std::mem::size_of::<[Ray; RAYS_PER_BATCH]>() as f32 / 1024.
    );
    println!(
        "[PathInfo; {}] = {} kbytes",
        RAYS_PER_BATCH,
        std::mem::size_of::<[PathInfo; RAYS_PER_BATCH]>() as f32 / 1024.
    );
    println!(
        "[Option<usize>; {}] = {} kbytes",
        RAYS_PER_BATCH,
        std::mem::size_of::<[Option<usize>; RAYS_PER_BATCH]>() as f32 / 1024.
    );

    let mut frames = 1;
    if args.bench {
        let warmup_start = Instant::now();

        let warmup_frames = 5;
        for _ in 0..warmup_frames {
            if args.singlethread {
                batches
                    .iter_mut()
                    .for_each(|batch| trace_batch(batch, &bvh, &triangles, &camera, &light_pos));
            } else {
                batches
                    .par_iter_mut()
                    .for_each(|batch| trace_batch(batch, &bvh, &triangles, &camera, &light_pos));
            }
        }

        let warmup_elapsed = warmup_start.elapsed();
        println!("warmup {:.2?} for {} frames", warmup_elapsed, warmup_frames);

        frames = (10. / (warmup_elapsed.as_secs_f32() / warmup_frames as f32)).ceil() as u32;
    }

    let time_start = Instant::now();

    for _ in 0..frames {
        if args.singlethread {
            batches
                .iter_mut()
                .for_each(|batch| trace_batch(batch, &bvh, &triangles, &camera, &light_pos));
        } else {
            batches
                .par_iter_mut()
                .for_each(|batch| trace_batch(batch, &bvh, &triangles, &camera, &light_pos));
        }
    }

    let elapsed = time_start.elapsed();
    println!(
        "{:.2?} for {} frames, {:.2?}/frame",
        elapsed,
        frames,
        elapsed / frames
    );

    let trace_stats = batches
        .iter()
        .fold(TraceStats::new(), |acc, x| acc + x.trace_stats);
    trace_stats.print();

    let image = RgbImage::from_fn(image_width, image_height, |x, y| {
        color_vec_to_rgb_norm(pixels[(y * image_width + x) as usize])
    });
    let now = Local::now();
    let _ = fs::create_dir("output");
    image
        .save(format!("output/{}.png", now.format("%Y%m%d_%H%M%S")))
        .unwrap();
}
