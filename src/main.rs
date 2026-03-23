mod aabb;
mod args;
mod brdf;
mod bvh;
mod camera;
mod integrators;
mod light;
mod math;
mod mesh;
mod ray;
mod scene;
mod trace_stats;
mod triangle_intersector;
mod triangle_opt;

use crate::args::{Args, TraceMode};
use crate::camera::Rect;
use crate::integrators::*;
use crate::math::{Vec3f, Vec4f};
use crate::scene::load_scene;
use crate::trace_stats::TraceStats;
use chrono::Local;
use clap::Parser as _;
use core::f32;
use image::RgbImage;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use simba::simd::{WideF32x4, WideF32x8};
use std::fs;
use std::sync::atomic::AtomicU32;
use std::time::Instant;

fn linear_to_gamma(v: Vec3f) -> Vec3f {
    v.map(|f| f.sqrt())
}

fn color_vec_to_rgb_norm_gamma(v: Vec4f) -> image::Rgb<u8> {
    let v = linear_to_gamma(v.xyz() / v.w) * 255.;
    image::Rgb([v.x as u8, v.y as u8, v.z as u8])
}

struct Tile<'a, R: Rng> {
    region: Rect,
    pixels: Vec<&'a mut Vec4f>,
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
    pixels: &'a mut [Vec4f],
    image_width: u32,
    image_height: u32,
    tile_size: u32,
    rng: &mut R,
) -> Vec<Tile<'a, StdRng>> {
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
            batches.push(Tile::new(rect, StdRng::from_rng(&mut *rng).unwrap()));
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
    let scene = load_scene(args.scene);

    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut pixels = vec![Vec4f::zeros(); (image_width * image_height) as usize];
    let mut tiles = generate_tiles(
        &mut pixels,
        image_width,
        image_height,
        args.tile_size,
        &mut rng,
    );

    println!(
        "{} {}x{} tiles, {} pixels/tile, {} samples/pixel, {} total samples/tile, max bounces {}, mode {:?}",
        tiles.len(),
        args.tile_size,
        args.tile_size,
        args.tile_size * args.tile_size,
        args.samples,
        args.tile_size * args.tile_size * args.samples,
        args.max_bounces,
        args.mode
    );

    let num_tiles = tiles.len();
    let num_tiles_completed = AtomicU32::new(0);

    let time_start = Instant::now();
    let trace_fn = |tile| {
        let trace_fn = match args.mode {
            TraceMode::Stream => integrate_tile_stream,
            TraceMode::StreamShadowImmediate => integrate_tile_stream_shadow_immediate,
            TraceMode::StreamCameraOnly => integrate_stream_camera_only,
            TraceMode::SingleRay => TileIntegrator1::integrate,
            TraceMode::Packet4 => TileIntegrator::<WideF32x4>::integrate,
            TraceMode::Packet8 => TileIntegrator::<WideF32x8>::integrate,
        };

        trace_fn(
            tile,
            (image_width, image_height),
            &scene,
            args.samples,
            args.max_bounces,
        );

        if args.progress {
            let n = num_tiles_completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            println!("{}/{} tiles completed", n + 1, num_tiles);
        }
    };

    if args.singlethread {
        tiles.iter_mut().for_each(trace_fn);
    } else {
        tiles.par_iter_mut().for_each(trace_fn);
    }

    let elapsed = time_start.elapsed();

    let trace_stats = tiles
        .iter()
        .fold(TraceStats::new(), |acc, x| acc + x.trace_stats);
    trace_stats.print();

    println!(
        "{:.2?} for {} samples/px, {:.2} samples/px/sec, {:.2?} MRays/sec, {:.2?} MSamples/sec",
        elapsed,
        args.samples,
        args.samples as f32 / elapsed.as_secs_f32(),
        trace_stats.total_rays() as f32 / 1_000_000. / elapsed.as_secs_f32(),
        (args.tile_size * args.tile_size * args.samples * num_tiles as u32) as f32
            / 1_000_000.
            / elapsed.as_secs_f32()
    );

    let image = RgbImage::from_fn(image_width, image_height, |x, y| {
        color_vec_to_rgb_norm_gamma(pixels[(y * image_width + x) as usize])
    });
    let now = Local::now();
    let _ = fs::create_dir("output");
    image
        .save(format!("output/{}.png", now.format("%Y%m%d_%H%M%S")))
        .unwrap();
}
