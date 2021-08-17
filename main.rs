mod aabb;
mod bvh;

use aabb::Aabb;
use bvh::Bvh;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;
use ultraviolet::{Vec2, Vec3, Vec3x4};
use wide::{f32x4, CmpGe, CmpGt, CmpLt};

const NUM_SUBSAMPLES: usize = 4;
const PACKET_SIZE: u32 = 8;

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

pub struct Frustum {
    normals: [Vec3; 4],
    offsets: [f32; 4],
}

impl Frustum {
    fn from_corner_rays(rays: &[Ray; 4]) -> Self {
        let mut normals: [Vec3; 4] = Default::default();
        let mut offsets: [f32; 4] = Default::default();

        for i in 0..4 {
            normals[i] = rays[i].direction * rays[(i + 1) % 4].direction;
            offsets[i] = rays[i].origin.dot(normals[i]);
        }

        Frustum { normals, offsets }
    }
}

#[derive(Clone, Copy)]
pub struct TraceResult<'a> {
    hit_dist: f32,
    object: Option<&'a Sphere>,
}

impl<'a> TraceResult<'a> {
    fn new() -> TraceResult<'a> {
        TraceResult {
            hit_dist: f32::INFINITY,
            object: None,
        }
    }

    fn add_hit(&mut self, hit_dist: f32, object: &'a Sphere) {
        if hit_dist < self.hit_dist {
            self.hit_dist = hit_dist;
            self.object = Some(object);
        }
    }
}

#[derive(Clone, Copy)]
pub struct TraceResultSimd<'a> {
    hit_dist: f32x4,
    object: [Option<&'a Sphere>; 4],
}

impl<'a> TraceResultSimd<'a> {
    fn new<'b>() -> TraceResultSimd<'b> {
        TraceResultSimd {
            hit_dist: f32x4::splat(f32::INFINITY),
            object: [None; 4],
        }
    }

    fn add_hit(&mut self, hit_dist: f32x4, object: &'a Sphere) {
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

pub struct Sphere {
    center: Vec3,
    centerx4: Vec3x4,
    radius: f32,
    radius2: f32,
    color: Vec3,
}

impl Sphere {
    fn new(center: Vec3, radius: f32, color: Vec3) -> Sphere {
        Sphere {
            center,
            centerx4: Vec3x4::splat(center),
            radius,
            radius2: radius * radius,
            color,
        }
    }

    fn aabb(&self) -> Aabb {
        let r = Vec3::broadcast(self.radius);
        Aabb {
            min: self.center - r,
            max: self.center + r,
        }
    }

    fn intersect(&self, ray_origin: &Vec3, ray_direction: &Vec3, backface: bool) -> Option<f32> {
        let r2 = self.radius2;
        let l = self.center - *ray_origin;
        let tca = l.dot(*ray_direction);
        let d2 = l.dot(l) - tca * tca;

        if d2 > r2 {
            None
        } else {
            let thc = (r2 - d2).sqrt();
            let t0 = tca - thc;
            let t1 = tca + thc;

            if t0 > 0. {
                Some(t0)
            } else if backface && t1 > 0. {
                Some(t1)
            } else {
                None
            }
        }
    }

    fn intersect_simd(&self, ray_origin: &Vec3x4, ray_direction: &Vec3x4, backface: bool) -> f32x4 {
        let l = self.centerx4 - *ray_origin;
        let tca = l.dot(*ray_direction);
        let d2 = l.dot(l) - tca * tca;

        let sqrt_valid = d2.cmp_lt(self.radius2);
        if sqrt_valid.any() {
            let thc = (self.radius2 - d2).sqrt();
            let t0 = tca - thc;
            let t0_valid = t0.cmp_gt(0.) & sqrt_valid;

            if backface {
                let t1 = tca + thc;
                let t1_valid = t1.cmp_gt(0.) & sqrt_valid;

                let t = t1_valid.blend(t1, f32x4::splat(f32::INFINITY));
                t0_valid.blend(t0, t)
            } else {
                t0_valid.blend(t0, f32x4::splat(f32::INFINITY))
            }
        } else {
            f32x4::splat(f32::INFINITY)
        }
    }
}

struct Scene {
    objects: Vec<Sphere>,
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

fn generate_random_scene() -> Scene {
    let mut rng = StdRng::seed_from_u64(646524362);
    let mut scene = Scene {
        objects: Vec::new(),
    };
    for _ in 0..20 {
        scene.objects.push(Sphere::new(
            Vec3::new(
                rng.gen_range(-10.0..10.0),
                rng.gen_range(-10.0..10.0),
                rng.gen_range(-10.0..10.0),
            ),
            rng.gen_range(0.5..2.0),
            Vec3::new(
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
            ),
        ));
    }

    scene
}

fn main() {
    let random_scene = true;
    let scene = if random_scene {
        generate_random_scene()
    } else {
        Scene {
            objects: vec![
                Sphere::new(Vec3::new(0., 0., 0.), 5., Vec3::new(0.8, 0.8, 0.8)),
                Sphere::new(Vec3::new(5., 0., -5.), 2., Vec3::new(0.8, 0.1, 0.1)),
            ],
        }
    };

    let bvh_start = Instant::now();
    let bvh = Bvh::build(&scene.objects);
    println!("bvh build {:.2?}", bvh_start.elapsed());

    let cam_pos = Vec3::new(0., 0., -30.);
    let cam_posx4 = Vec3x4::splat(cam_pos);
    let light_pos = Vec3::new(10., 10., -20.);
    let light_posx4 = Vec3x4::splat(light_pos);
    let image_width = 1920;
    let image_height = 1080;
    let mut image = image::RgbImage::new(image_width, image_height);

    let camera_rays = generate_camera_rays(image_width, image_height, 90.);
    let get_camera_ray_index = |x: u32, y: u32| {
        y as usize * image_width as usize * NUM_SUBSAMPLES + x as usize * NUM_SUBSAMPLES
    };

    let mut pixels: Vec<_> = image.enumerate_pixels_mut().collect();

    pixels.sort_by_key(|(x, y, _)| (y / PACKET_SIZE, x / PACKET_SIZE, *y, *x));

    let mut packets: Vec<_> = pixels
        .chunks_mut((PACKET_SIZE * PACKET_SIZE) as usize)
        .map(|pixels| {
            let mut rays = Vec::with_capacity(pixels.len() * NUM_SUBSAMPLES);
            for (x, y, _) in &*pixels {
                let rays_index = get_camera_ray_index(*x, *y);
                for i in 0..NUM_SUBSAMPLES {
                    rays.push(Ray::new(&cam_pos, &camera_rays[rays_index + i]));
                }
            }

            let mut min = Vec2::broadcast(f32::INFINITY);
            let mut max = Vec2::broadcast(f32::NEG_INFINITY);
            for r in &rays {
                let uv = r.direction.xy() / r.direction.z;
                min = min.min_by_component(uv);
                max = max.max_by_component(uv);
            }

            let frustum = Frustum::from_corner_rays(&[
                Ray::new(&cam_pos, &Vec3::new(min.x, min.y, 1.).normalized()),
                Ray::new(&cam_pos, &Vec3::new(max.x, min.y, 1.).normalized()),
                Ray::new(&cam_pos, &Vec3::new(max.x, max.y, 1.).normalized()),
                Ray::new(&cam_pos, &Vec3::new(min.x, max.y, 1.).normalized()),
            ]);

            (pixels, rays, frustum)
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

    let frames = 500;
    for _ in 0..frames {
        packets.par_iter_mut().for_each(|(pixels, rays, frustum)| {
            let mut trace_results =
                [TraceResult::new(); PACKET_SIZE as usize * PACKET_SIZE as usize * NUM_SUBSAMPLES];
            bvh.trace_packet(rays, frustum, &mut trace_results);

            for (i, (x, y, pixel)) in pixels.iter_mut().enumerate() {
                let closest_hit = f32x4::from([
                    trace_results[(i * 4)].hit_dist,
                    trace_results[(i * 4 + 1)].hit_dist,
                    trace_results[(i * 4 + 2)].hit_dist,
                    trace_results[(i * 4 + 3)].hit_dist,
                ]);
                let closest_obj = [
                    trace_results[(i * 4)].object,
                    trace_results[(i * 4 + 1)].object,
                    trace_results[(i * 4 + 2)].object,
                    trace_results[(i * 4 + 3)].object,
                ];
                // let closest_hit = f32x4::splat(f32::INFINITY);
                // let closest_obj: [Option<&Sphere>; NUM_SUBSAMPLES] = [None; NUM_SUBSAMPLES];

                if closest_hit.cmp_lt(f32::INFINITY).none() {
                    continue;
                }

                let centers = Vec3x4::from([
                    if let Some(o) = closest_obj[0] {
                        o.center
                    } else {
                        Vec3::zero()
                    },
                    if let Some(o) = closest_obj[1] {
                        o.center
                    } else {
                        Vec3::zero()
                    },
                    if let Some(o) = closest_obj[2] {
                        o.center
                    } else {
                        Vec3::zero()
                    },
                    if let Some(o) = closest_obj[3] {
                        o.center
                    } else {
                        Vec3::zero()
                    },
                ]);

                let rays = Vec3x4::from([
                    rays[(i * 4)].direction,
                    rays[(i * 4) + 1].direction,
                    rays[(i * 4) + 2].direction,
                    rays[(i * 4) + 3].direction,
                ]);

                let hit_pos = cam_posx4 + rays * closest_hit;

                let shadow_ray = (light_posx4 - hit_pos).normalized();
                let TraceResultSimd {
                    hit_dist: shadow_hit,
                    ..
                } = bvh.trace(&hit_pos, &shadow_ray);

                if shadow_hit.cmp_lt(f32::INFINITY).all() {
                    continue;
                }

                let shadow_hit: [f32; 4] = shadow_hit.into();

                let light_dir = (light_posx4 - hit_pos).normalized();
                let normal = (hit_pos - centers).normalized();
                let ndl = light_dir.dot(normal);
                let ndl: [f32; 4] = ndl.into();
                let mut color = Vec3::zero();
                for i in 0..4 {
                    if let Some(o) = closest_obj[i] {
                        if shadow_hit[i] >= f32::INFINITY {
                            color += o.color * ndl[i] / 4.;
                        }
                    }
                }

                **pixel = color_vec_to_rgb(color);
            }
        });
    }

    let elapsed = time_start.elapsed();
    println!(
        "{:.2?} for {} frames, {:.2?}/frame",
        elapsed,
        frames,
        elapsed / frames
    );

    image.save("output.png").unwrap();
}
