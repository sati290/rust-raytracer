mod aabb;
mod bvh;
mod frustum;

use aabb::Aabb;
use bvh::{Bvh, TraceStats};
use frustum::Frustum;
use obj::Obj;
use rayon::prelude::*;
use std::time::Instant;
use ultraviolet::{Isometry3, Mat3, Rotor3, Vec2, Vec3, Vec3x4};
use wide::{f32x4, CmpGe, CmpLe, CmpLt};

const NUM_SUBSAMPLES: usize = 4;
const PACKET_SIZE: u32 = 16;

pub struct Ray {
    origin: Vec3,
    direction: Vec3,
    direction_recip: Vec3,
}

impl Ray {
    fn new(origin: &Vec3, direction: &Vec3) -> Self {
        Ray {
            origin: *origin,
            direction: *direction,
            direction_recip: Vec3::one() / *direction,
        }
    }
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

pub struct Triangle {
    verts: [Vec3; 3],
    normal: Vec3,
}

impl Triangle {
    pub fn new(verts: [Vec3; 3]) -> Self {
        let v0v1 = verts[1] - verts[0];
        let v0v2 = verts[2] - verts[0];
        let normal = v0v2.cross(v0v1).normalized();

        Triangle { verts, normal }
    }

    pub fn aabb(&self) -> Aabb {
        let mut aabb = Aabb::empty();

        for v in self.verts {
            aabb.grow_mut(v);
        }

        aabb
    }

    pub fn centroid(&self) -> Vec3 {
        (self.verts[0] + self.verts[1] + self.verts[2]) / 3.
    }

    fn intersect<const B: bool>(&self, ray_origin: &Vec3, ray_direction: &Vec3) -> Option<f32> {
        let v0v1 = self.verts[1] - self.verts[0];
        let v0v2 = self.verts[2] - self.verts[0];
        let pvec = ray_direction.cross(v0v1);
        let det = v0v2.dot(pvec);

        let epsilon = 0.0000001;
        if det < epsilon {
            return None;
        }

        let inv_det = 1. / det;

        let tvec = *ray_origin - self.verts[0];
        let u = tvec.dot(pvec) * inv_det;
        if !(0. ..=1.).contains(&u) {
            return None;
        }

        let qvec = tvec.cross(v0v2);
        let v = ray_direction.dot(qvec) * inv_det;
        if !(0. ..=1.).contains(&v) {
            return None;
        }

        let t = v0v1.dot(qvec) * inv_det;

        Some(t)
    }

    fn intersect_simd<const B: bool>(&self, ray_origin: &Vec3x4, ray_direction: &Vec3x4) -> f32x4 {
        let v0v1 = Vec3x4::splat(self.verts[1] - self.verts[0]);
        let v0v2 = Vec3x4::splat(self.verts[2] - self.verts[0]);
        let pvec = ray_direction.cross(v0v1);
        let det = v0v2.dot(pvec);

        let epsilon = f32x4::splat(0.0000001);
        let det_valid = det.cmp_ge(epsilon);

        let inv_det = 1. / det;

        let tvec = *ray_origin - Vec3x4::splat(self.verts[0]);
        let u = tvec.dot(pvec) * inv_det;
        let u_valid = u.cmp_ge(0.) & u.cmp_le(1.);

        let qvec = tvec.cross(v0v2);
        let v = ray_direction.dot(qvec) * inv_det;
        let v_valid = v.cmp_ge(0.) & v.cmp_le(1.);

        let t = v0v1.dot(qvec) * inv_det;

        let t_valid = det_valid & u_valid & v_valid;
        t_valid.blend(t, f32x4::splat(f32::INFINITY))
    }
}

struct RayPacket<'a> {
    pixels: &'a mut [(u32, u32, &'a mut image::Rgb<u8>)],
    rays: Vec<Ray>,
    frustum: Frustum,
    trace_stats: TraceStats,
    // shadow_rays_total: u32,
    // shadow_rays_active: u32,
}

impl<'a> RayPacket<'a> {
    fn new(pixels: &'a mut [(u32, u32, &'a mut image::Rgb<u8>)], rays: Vec<Ray>) -> Self {
        let mut min = Vec2::broadcast(f32::INFINITY);
        let mut max = Vec2::broadcast(f32::NEG_INFINITY);
        for r in &rays {
            let uv = r.direction.xy() / r.direction.z;
            min = min.min_by_component(uv);
            max = max.max_by_component(uv);
        }

        let corner_rays = [
            Ray::new(&Vec3::zero(), &Vec3::new(min.x, min.y, 1.)),
            Ray::new(&Vec3::zero(), &Vec3::new(min.x, max.y, 1.)),
            Ray::new(&Vec3::zero(), &Vec3::new(max.x, max.y, 1.)),
            Ray::new(&Vec3::zero(), &Vec3::new(max.x, min.y, 1.)),
        ];

        let frustum = Frustum::from_corner_rays(&corner_rays);

        RayPacket {
            pixels,
            rays,
            frustum,
            trace_stats: TraceStats::new(),
            // shadow_rays_total: 0,
            // shadow_rays_active: 0,
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

fn trace_packet<'a>(
    packet: &mut RayPacket<'a>,
    bvh: &'a Bvh,
    cam_pos: &Vec3,
    cam_transform: &Isometry3,
    light_pos: &Vec3,
) {
    let cam_posx4 = Vec3x4::splat(*cam_pos);
    let light_posx4 = Vec3x4::splat(*light_pos);

    let transformed_rays: Vec<_> = packet
        .rays
        .iter()
        .map(|r| Ray::new(cam_pos, &(cam_transform.rotation * r.direction)))
        .collect();

    let frustum = cam_transform.into_homogeneous_matrix() * packet.frustum;

    let mut trace_results =
        [TraceResult::new(); PACKET_SIZE as usize * PACKET_SIZE as usize * NUM_SUBSAMPLES];
    bvh.trace_packet(
        &transformed_rays,
        &frustum,
        &mut trace_results,
        &mut packet.trace_stats,
    );

    for ((_x, _y, pixel), (rays, results)) in packet.pixels.iter_mut().zip(
        packet
            .rays
            .chunks_exact(NUM_SUBSAMPLES)
            .zip(trace_results.chunks_exact(NUM_SUBSAMPLES)),
    ) {
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
            rays[0].direction,
            rays[1].direction,
            rays[2].direction,
            rays[3].direction,
        ]);

        let hit_pos = cam_posx4 + rays * closest_hit;

        let shadow_ray = (light_posx4 - hit_pos).normalized();
        //let shadow_hit = bvh.trace_shadow(&hit_pos, &shadow_ray, hit_mask);
        let shadow_hit = 0;
        // packet.shadow_rays_total += 4;
        // packet.shadow_rays_active += closest_hit.cmp_lt(f32::INFINITY).move_mask().count_ones();

        if shadow_hit == 0b1111 {
            continue;
        }
        let closest_obj = [
            results[0].object,
            results[1].object,
            results[2].object,
            results[3].object,
        ];

        let normal = Vec3x4::from([
            closest_obj[0].map_or(Vec3::zero(), |o| o.normal),
            closest_obj[1].map_or(Vec3::zero(), |o| o.normal),
            closest_obj[2].map_or(Vec3::zero(), |o| o.normal),
            closest_obj[3].map_or(Vec3::zero(), |o| o.normal),
        ]);

        let light_dir = (light_posx4 - hit_pos).normalized();
        let ndl = light_dir.dot(normal);
        let ndl: [f32; 4] = ndl.into();
        let mut color = Vec3::zero();
        for i in 0..4 {
            if let Some(o) = closest_obj[i] {
                if shadow_hit & 1 << i == 0 {
                    color += Vec3::one() * ndl[i] / 4.;
                }
            }
        }

        **pixel = color_vec_to_rgb(color);
    }
}

fn load_scene() -> Vec<Triangle> {
    let load_start = Instant::now();

    let obj = Obj::load("./asian_dragon_obj/asian_dragon.obj").unwrap();

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
    let mut image = image::RgbImage::new(image_width, image_height);

    let camera_transform = Isometry3::new(cam_pos, look_at(&cam_pos, &cam_target));

    let camera_rays = generate_camera_rays(image_width, image_height, 90.);
    let get_camera_ray_index = |x: u32, y: u32| {
        y as usize * image_width as usize * NUM_SUBSAMPLES + x as usize * NUM_SUBSAMPLES
    };

    let mut pixels: Vec<_> = image.enumerate_pixels_mut().collect();

    pixels.sort_by_key(|(x, y, _)| (y / PACKET_SIZE, x / PACKET_SIZE, *y, *x));

    let mut packets: Vec<_> = pixels
        .chunks_mut((PACKET_SIZE * PACKET_SIZE) as usize)
        .map(|pixels| {
            let rays = pixels
                .iter()
                .flat_map(|(x, y, _)| {
                    let camera_rays = &camera_rays;
                    let rays_index = get_camera_ray_index(*x, *y);
                    (0..NUM_SUBSAMPLES)
                        .map(move |i| Ray::new(&Vec3::zero(), &camera_rays[rays_index + i]))
                })
                .collect();

            RayPacket::new(pixels, rays)
        })
        .collect();

    println!(
        "{} {}x{} packets, {} rays/packet",
        packets.len(),
        PACKET_SIZE,
        PACKET_SIZE,
        PACKET_SIZE * PACKET_SIZE * NUM_SUBSAMPLES as u32
    );

    let time_start = Instant::now();

    let frames = 5;
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

    // let shadow_rays_total = packets.iter().fold(0, |acc, x| acc + x.shadow_rays_total);
    // let shadow_rays_active = packets.iter().fold(0, |acc, x| acc + x.shadow_rays_active);
    // println!("shadow trace simd utilization {}%", shadow_rays_active as f32 / shadow_rays_total as f32 * 100.);
    let trace_stats = packets
        .iter()
        .fold(TraceStats::new(), |acc, x| acc + x.trace_stats);
    println!("{:?}", trace_stats);

    image.save("output.png").unwrap();
}
