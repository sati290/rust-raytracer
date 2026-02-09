mod aabb;
mod bvh;
mod triangle;

use image::RgbImage;
use triangle::Triangle;
use bvh::{Bvh, TraceStats};
use obj::Obj;
use rayon::prelude::*;
use std::time::Instant;
use ultraviolet::{Isometry3, Mat3, Rotor3, Vec2, Vec3, Vec3x4, Vec4};
use wide::{f32x4, CmpGe, CmpLt};

const NUM_SUBSAMPLES: usize = 4;
const PACKET_SIZE: u32 = 32;

#[repr(C, align(16))]
pub struct Ray {
    origin_near: Vec4,         // x, y, z, near
    direction_recip_far: Vec4, // x, y, z, far
    direction: Vec4,
}

impl Ray {
    fn new(origin: &Vec3, direction: &Vec3) -> Self {
        let dir_recip = Vec3::one() / *direction;
        Ray {
            origin_near: Vec4::new(origin.x, origin.y, origin.z, 0.),
            direction: Vec4::from(*direction),
            direction_recip_far: Vec4::new(dir_recip.x, dir_recip.y, dir_recip.z, f32::INFINITY),
        }
    }
}

struct RayInfo {
    contribution: Vec3,
    destination_idx: usize,
}

#[derive(Clone, Copy)]
pub struct TraceResult<'a> {
    hit_dist: f32,
    object: Option<&'a Triangle>,
}

impl<'a> TraceResult<'a> {
    fn new() -> TraceResult<'a> {
        TraceResult {
            hit_dist: f32::INFINITY,
            object: None,
        }
    }

    fn add_hit(&mut self, hit_dist: f32, object: &'a Triangle) {
        if hit_dist < self.hit_dist {
            self.hit_dist = hit_dist;
            self.object = Some(object);
        }
    }
}

#[derive(Clone, Copy)]
pub struct TraceResultSimd<'a> {
    hit_dist: f32x4,
    object: [Option<&'a Triangle>; 4],
}

impl<'a> TraceResultSimd<'a> {
    fn new<'b>() -> TraceResultSimd<'b> {
        TraceResultSimd {
            hit_dist: f32x4::splat(f32::INFINITY),
            object: [None; 4],
        }
    }

    fn add_hit(&mut self, hit_dist: f32x4, object: &'a Triangle) {
        let closest = hit_dist.cmp_lt(self.hit_dist);
        self.hit_dist = self.hit_dist.min(hit_dist);

        let closest_mask = closest.move_mask();
        for (i, obj) in self.object.iter_mut().enumerate() {
            if closest_mask & 1 << i != 0 {
                *obj = Some(object);
            }
        }
    }
}

struct RayPacket<'a> {
    pixels: &'a mut [(usize, Vec3)],
    ray_directions: Vec<Vec3>,
    ray_infos: Vec<RayInfo>,
    trace_stats: TraceStats,
}

impl<'a> RayPacket<'a> {
    fn new(
        pixels: &'a mut [(usize, Vec3)],
        ray_directions: Vec<Vec3>,
        ray_infos: Vec<RayInfo>,
    ) -> Self {
        RayPacket {
            pixels,
            ray_directions,
            ray_infos,
            trace_stats: TraceStats::new(),
        }
    }
}

fn color_vec_to_rgb(v: Vec3) -> image::Rgb<u8> {
    image::Rgb([(v.x * 255.) as u8, (v.y * 255.) as u8, (v.z * 255.) as u8])
}

fn generate_camera_rays(image_width: u32, image_height: u32, horiz_fog_deg: f32) -> Vec<Vec3> {
    let subpixels = [
        Vec2::new(5. / 8., 1. / 8.),
        Vec2::new(1. / 8., 3. / 8.),
        Vec2::new(7. / 8., 5. / 8.),
        Vec2::new(3. / 8., 7. / 8.),
    ];

    let plane_dist = (horiz_fog_deg.to_radians() / 2.).tan();
    let image_dims_recip = Vec2::new(
        1. / image_width as f32,
        1. / image_width /* width used here to handle aspect */ as f32,
    );

    let mut rays =
        Vec::<Vec3>::with_capacity(image_width as usize * image_height as usize * subpixels.len());
    for y in 0..image_height {
        for x in 0..image_width {
            let px = Vec2::new(x as f32, (image_height - y) as f32);
            let px = px - Vec2::new(image_width as f32 / 2., image_height as f32 / 2.);
            for sp in subpixels {
                let px = (px + sp) * image_dims_recip;
                let ray = Vec3::new(px.x, px.y, plane_dist).normalized();
                rays.push(ray);
            }
        }
    }

    rays
}

fn brdf(dir_in: Vec3x4, _dir_out: Vec3x4, normal: Vec3x4) -> Vec3x4 {
    let ndotl = normal.dot(dir_in);
    return Vec3x4::one() * ndotl.max(f32x4::ZERO);
}

fn trace_packet<'a>(
    packet: &mut RayPacket<'a>,
    bvh: &'a Bvh,
    cam_pos: &Vec3,
    cam_transform: &Isometry3,
    light_pos: &Vec3,
) {
    packet.pixels.iter_mut().for_each(|(_, color)| *color = Vec3::zero());

    let cam_posx4 = Vec3x4::splat(*cam_pos);
    let light_posx4 = Vec3x4::splat(*light_pos);

    let mut transformed_rays: Vec<_> = packet
        .ray_directions
        .iter()
        .map(|r| Ray::new(cam_pos, &(cam_transform.rotation * *r)))
        .collect();

    let mut trace_results =
        [TraceResult::new(); PACKET_SIZE as usize * PACKET_SIZE as usize * NUM_SUBSAMPLES];
    bvh.trace_stream(
        &mut transformed_rays,
        &mut trace_results,
        &mut packet.trace_stats,
    );

    let mut shadow_rays = Vec::with_capacity(transformed_rays.len());
    let mut shadow_ray_infos = Vec::with_capacity(transformed_rays.len());
    
    for ((ray_infos, rays), results) in packet.ray_infos.chunks_exact_mut(4)
        .zip(transformed_rays.chunks_exact(4))
        .zip(trace_results.chunks_exact(4))
    {
        let closest_hit = f32x4::from([
            results[0].hit_dist,
            results[1].hit_dist,
            results[2].hit_dist,
            results[3].hit_dist,
        ]);
        
        let hit_mask = closest_hit.cmp_lt(f32::INFINITY).move_mask();
        if hit_mask == 0 {
            continue;
        }

        let rays = Vec3x4::from([
            rays[0].direction.xyz(),
            rays[1].direction.xyz(),
            rays[2].direction.xyz(),
            rays[3].direction.xyz(),
        ]);

        let hit_pos = cam_posx4 + rays * closest_hit;
        let shadow_ray_dirs = (light_posx4 - hit_pos).normalized();

        let normal = Vec3x4::from([
            results[0].object.map_or(Vec3::zero(), |o| o.normal),
            results[1].object.map_or(Vec3::zero(), |o| o.normal),
            results[2].object.map_or(Vec3::zero(), |o| o.normal),
            results[3].object.map_or(Vec3::zero(), |o| o.normal),
        ]);

        let ratio = brdf(shadow_ray_dirs, -rays, normal);
        let hit_pos: [Vec3; 4] = hit_pos.into();
        let shadow_ray_dirs: [Vec3; 4] = shadow_ray_dirs.into();
        let ratio: [Vec3; 4] = ratio.into();
        for i in 0..4 {
            if hit_mask & (1 << i) != 0 {
                shadow_rays.push(Ray::new(&hit_pos[i], &shadow_ray_dirs[i]));
                shadow_ray_infos.push(RayInfo {
                    contribution: ray_infos[i].contribution * ratio[i],
                    destination_idx: ray_infos[i].destination_idx,
                });
            }
        }
    }

    trace_results = [TraceResult::new(); PACKET_SIZE as usize * PACKET_SIZE as usize * NUM_SUBSAMPLES];
    bvh.trace_stream(&mut shadow_rays, &mut trace_results, &mut packet.trace_stats);

    for (ray_infos, results) in shadow_ray_infos.chunks_exact_mut(4).zip(trace_results.chunks_exact(4))
    {
        let closest_hit = f32x4::from([
            results[0].hit_dist,
            results[1].hit_dist,
            results[2].hit_dist,
            results[3].hit_dist,
        ]);

        let hit_mask = closest_hit.cmp_lt(f32::INFINITY).move_mask();

        if hit_mask == 0b1111 {
            continue;
        }

        for i in 0..4 {
            if hit_mask & (1 << i) == 0 {
                packet.pixels[ray_infos[i].destination_idx].1 += ray_infos[i].contribution;
            }
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

fn look_at(eye: &Vec3, target: &Vec3) -> Rotor3 {
    let at = (*target - *eye).normalized();
    let up = Vec3::unit_y();
    let side = up.cross(at);
    let rup = at.cross(side);
    let mat = Mat3::new(side, rup, at);

    mat.into_rotor3()
}

fn main() {
    let triangles = load_scene();
    let bvh_start = Instant::now();
    let bvh = Bvh::build(&triangles);
    println!("bvh build {:.2?}", bvh_start.elapsed());

    let cam_pos = Vec3::new(0.6, 0.25, -1.).normalized() * 3000.;
    let cam_target = Vec3::new(0., 350., 0.);
    let light_pos = Vec3::new(5000., 5000., -10000.);
    let image_width = 1920;
    let image_height = 1080;

    let camera_transform = Isometry3::new(cam_pos, look_at(&cam_pos, &cam_target));

    let camera_rays = generate_camera_rays(image_width, image_height, 90.);

    let mut pixels: Vec<_> = (0..(image_width * image_height) as usize).map(|i| (i, Vec3::zero())).collect();

    let mut packets: Vec<_> = pixels
        .chunks_mut((PACKET_SIZE * PACKET_SIZE) as usize)
        .map(|pixels| {
            let rays = pixels
                .iter()
                .flat_map(|(i, _)| {
                    let camera_rays = &camera_rays;
                    let rays_index = i * NUM_SUBSAMPLES;//get_camera_ray_index(*x, *y);
                    (0..NUM_SUBSAMPLES).map(move |i| camera_rays[rays_index + i])
                })
                .collect();
            
            let ray_infos = (0..pixels.len())
                .flat_map(|i|
                    (0..NUM_SUBSAMPLES).map(move |_| RayInfo {
                        contribution: Vec3::broadcast(1. / (NUM_SUBSAMPLES as f32)),
                        destination_idx: i,
                    })
                )
                .collect();

            RayPacket::new(pixels, rays, ray_infos)
        })
        .collect();

    println!(
        "{} {}x{} packets, {} rays/packet",
        packets.len(),
        PACKET_SIZE,
        PACKET_SIZE,
        PACKET_SIZE * PACKET_SIZE * NUM_SUBSAMPLES as u32
    );

    let warmup_start = Instant::now();

    let warmup_frames = 5;
    for _ in 0..warmup_frames {
        packets
            .par_iter_mut()
            .for_each(|packet| trace_packet(packet, &bvh, &cam_pos, &camera_transform, &light_pos));
    }

    let warmup_elapsed = warmup_start.elapsed();
    println!("warmup {:.2?} for {} frames", warmup_elapsed, warmup_frames);

    let frames = (15. / (warmup_elapsed.as_secs_f32() / warmup_frames as f32)).ceil() as u32;
    let time_start = Instant::now();

    for _ in 0..frames {
        packets
            .par_iter_mut()
            .for_each(|packet| trace_packet(packet, &bvh, &cam_pos, &camera_transform, &light_pos));
    }

    let elapsed = time_start.elapsed();
    println!(
        "{:.2?} for {} frames, {:.2?}/frame",
        elapsed,
        frames,
        elapsed / frames
    );

    let trace_stats = packets
        .iter()
        .fold(TraceStats::new(), |acc, x| acc + x.trace_stats);
    println!("{:?}", trace_stats);

    let image = RgbImage::from_fn(image_width, image_height, |x, y| color_vec_to_rgb(pixels[(y * image_width + x) as usize].1));
    image.save("output.png").unwrap();
}
