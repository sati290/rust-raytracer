mod aabb;
mod bvh;
mod triangle;
mod triangle_opt;
mod camera;

use image::RgbImage;
use obj::Obj;
use rayon::prelude::*;
use std::{fs};
use std::time::Instant;
use ultraviolet::{Vec3, Vec3x4, Vec4};
use wide::{f32x4, CmpGe};
use chrono::{Local};
use crate::triangle::Triangle;
use crate::camera::{Camera, Rect};
use crate::bvh::{Bvh, TraceStats};


const NUM_SUBSAMPLES: usize = 4;
const BATCH_SIZE: u32 = 64;

#[repr(C, align(16))]
pub struct Ray {
    origin_near: Vec4,         // x, y, z, near
    direction_recip_far: Vec4, // x, y, z, far
    direction: Vec4,
}

impl Ray {
    #[must_use]
    fn new(origin: &Vec3, direction: &Vec3) -> Self {
        let dir_recip = Vec3::one() / *direction;
        Ray {
            origin_near: Vec4::new(origin.x, origin.y, origin.z, 0.),
            direction: Vec4::from(*direction),
            direction_recip_far: Vec4::new(dir_recip.x, dir_recip.y, dir_recip.z, f32::INFINITY),
        }
    }

    #[must_use]
    fn _is_hit(&self) -> bool {
        self.direction_recip_far.w < f32::INFINITY
    }

    #[must_use]
    fn hit_dist(&self) -> f32 {
        self.direction_recip_far.w
    }

    #[must_use]
    fn hit_pos(&self) -> Vec3 {
        self.origin_near.xyz() + self.direction.xyz() * self.hit_dist()
    }
}

pub struct PathInfo {
    contribution: Vec3,
    destination_idx: usize,
}

fn color_vec_to_rgb(v: Vec3) -> image::Rgb<u8> {
    image::Rgb([(v.x * 255.) as u8, (v.y * 255.) as u8, (v.z * 255.) as u8])
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

fn trace_batch(
    batch: &mut Batch,
    bvh: &Bvh,
    objects: &[Triangle],
    camera: &Camera,
    light_pos: &Vec3,
) {
    for p in batch.pixels.iter_mut() {
        **p = Vec3::zero();
    }

    let mut rays = camera.generate_rays(&batch.region);
    let mut ray_infos = Vec::with_capacity(batch.pixels.len() * NUM_SUBSAMPLES);
    for i in 0..rays.len() {
        ray_infos.push(PathInfo {
            contribution: Vec3::one() / NUM_SUBSAMPLES as f32,
            destination_idx: i / NUM_SUBSAMPLES,
        });
    }

    let mut hit_objects = vec![None; rays.len()];
    bvh.trace_stream(
        &mut rays,
        &mut hit_objects,
        &mut batch.trace_stats,
    );

    let mut new_rays_len = 0;
    for (i, hit_obj_idx) in hit_objects.iter().enumerate() {
        if let &Some(hit_obj_idx) = hit_obj_idx {
            let ray = &rays[i];
            let path_info = &ray_infos[i];
            let hit_obj = &objects[hit_obj_idx];
            let hit_pos = ray.hit_pos();
            let shadow_ray_dir = (*light_pos - hit_pos).normalized();
            let brdf = brdf(shadow_ray_dir, -ray.direction.xyz(), hit_obj.normal);
            rays[new_rays_len] = Ray::new(&hit_pos, &shadow_ray_dir);
            ray_infos[new_rays_len] = PathInfo { contribution: path_info.contribution * brdf, ..*path_info };
            new_rays_len += 1;
        }
    }

    rays.truncate(new_rays_len);
    ray_infos.truncate(new_rays_len);

    hit_objects.clear();
    hit_objects.resize(rays.len(), None);
    bvh.trace_stream(&mut rays, &mut hit_objects, &mut batch.trace_stats);

    for (ray_info, ray) in ray_infos.into_iter().zip(rays.into_iter()) {
        if ray.hit_dist() == f32::INFINITY {
            *batch.pixels[ray_info.destination_idx] += ray_info.contribution;
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

struct Batch<'a> {
    region: Rect,
    pixels: Vec<&'a mut Vec3>,
    trace_stats: TraceStats,
}

impl Batch<'_> {
    fn new(region: Rect) -> Self {
        Batch {
            region,
            pixels: Vec::new(),
            trace_stats: TraceStats::new(),
        }
    }
}

fn generate_batches(pixels: &mut [Vec3], image_width: u32, image_height: u32, batch_size: u32) -> Vec<Batch<'_>> {
    let num_regions_x = image_width.div_ceil(batch_size);
    let num_regions_y = image_height.div_ceil(batch_size);

    let mut batches = Vec::with_capacity((num_regions_x * num_regions_y) as usize);
    for region_y in 0..num_regions_y {
        for region_x in 0..num_regions_x {
            let x = region_x * batch_size;
            let y = region_y * batch_size;
            let width = batch_size.min(image_width - x);
            let height = batch_size.min(image_height - y);
            let rect = Rect { x, y, width, height };
            batches.push(Batch::new(rect));
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
    let triangles = load_scene();
    let bvh_start = Instant::now();
    let bvh = Bvh::build(&triangles);
    println!("bvh build {:.2?}", bvh_start.elapsed());

    let cam_pos = Vec3::new(0.6, 0.25, -1.).normalized() * 2500.;
    let cam_target = Vec3::new(0., 350., 0.);
    let light_pos = Vec3::new(5000., 5000., -10000.);
    let image_width = 1920;
    let image_height = 1080;

    let camera = Camera::new(cam_pos, cam_target, Vec3::new(0., 1., 0.), 60., image_width, image_height);

    let mut pixels = vec![Vec3::zero(); (image_width * image_height) as usize];
    let mut batches = generate_batches(&mut pixels, image_width, image_height, BATCH_SIZE);

    println!(
        "{} {}x{} batches, {} rays/batch",
        batches.len(),
        BATCH_SIZE,
        BATCH_SIZE,
        BATCH_SIZE * BATCH_SIZE * NUM_SUBSAMPLES as u32
    );

    let warmup_start = Instant::now();

    let warmup_frames = 5;
    for _ in 0..warmup_frames {
        batches
            .par_iter_mut()
            .for_each(|batch| trace_batch(batch, &bvh, &triangles, &camera, &light_pos));
    }

    let warmup_elapsed = warmup_start.elapsed();
    println!("warmup {:.2?} for {} frames", warmup_elapsed, warmup_frames);

    let frames = (15. / (warmup_elapsed.as_secs_f32() / warmup_frames as f32)).ceil() as u32;
    let time_start = Instant::now();

    for _ in 0..frames {
        batches
            .par_iter_mut()
            .for_each(|batch| trace_batch(batch, &bvh, &triangles, &camera, &light_pos));
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
    println!("{:?}", trace_stats);
    println!("avg {} rays {} objs per leaf visit", trace_stats.leaf_rays / trace_stats.leaf_visit, trace_stats.leaf_objs / trace_stats.leaf_visit);

    let image = RgbImage::from_fn(image_width, image_height, |x, y| color_vec_to_rgb(pixels[(y * image_width + x) as usize]));
    let now = Local::now();
    let _ = fs::create_dir("output");
    image.save(format!("output/{}.png", now.format("%Y%m%d_%H%M%S"))).unwrap();
}
