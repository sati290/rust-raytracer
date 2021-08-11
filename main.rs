mod aabb;
mod bvh;

use aabb::Aabb;
use bvh::Bvh;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;
use ultraviolet::{Vec2, Vec3, Vec3x4};
use wide::{f32x4, CmpGt, CmpLt};

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

    fn intersectx4(&self, ray_origin: &Vec3x4, ray_direction: &Vec3x4, backface: bool) -> f32x4 {
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

                let t = t1_valid.blend(t1, f32x4::splat(f32::MAX));
                t0_valid.blend(t0, t)
            } else {
                t0_valid.blend(t0, f32x4::splat(f32::MAX))
            }
        } else {
            f32x4::splat(f32::MAX)
        }
    }
}

struct Scene {
    objects: Vec<Sphere>,
}

impl Scene {
    fn tracex4(
        &self,
        origin: &Vec3x4,
        direction: &Vec3x4,
        backface: bool,
    ) -> (f32x4, [Option<&Sphere>; 4]) {
        let mut closest_hit = f32x4::splat(f32::MAX);
        let mut closest_obj: [Option<&Sphere>; 4] = [None; 4];
        for o in &self.objects {
            let hit = o.intersectx4(origin, direction, backface);
            let closest = hit.cmp_lt(closest_hit);
            closest_hit = closest.blend(hit, closest_hit);

            let closest_mask = closest.move_mask();
            for (i, obj) in closest_obj.iter_mut().enumerate() {
                if closest_mask & 1 << i != 0 {
                    *obj = Some(o);
                }
            }
        }

        (closest_hit, closest_obj)
    }
}

fn color_vec_to_rgb(v: Vec3) -> image::Rgb<u8> {
    image::Rgb([(v.x * 255.) as u8, (v.y * 255.) as u8, (v.z * 255.) as u8])
}

fn generate_camera_rays(
    image_width: u32,
    image_height: u32,
    horiz_fog_deg: f32,
) -> Vec<(u32, u32, [Vec3; 4])> {
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
        Vec::<(u32, u32, [Vec3; 4])>::with_capacity((image_width * image_height) as usize);
    for y in 0..image_height {
        for x in 0..image_width {
            let mut sp_rays = [Vec3::zero(); 4];

            let px = Vec2::new(x as f32, (image_height - y) as f32);
            let px = px - Vec2::new(image_width as f32 / 2., image_height as f32 / 2.);
            for i in 0..4 {
                let px = (px + subpixels[i]) * image_dims_recip;
                let ray = Vec3::new(px.x, px.y, plane_dist).normalized();
                sp_rays[i] = ray;
            }

            rays.push((x, y, sp_rays));
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

    let mut camera_rays = generate_camera_rays(image_width, image_height, 90.);
    camera_rays.sort_by_key(|(x, y, _)| (*x, *y));

    let mut rt_jobs: Vec<_> = image
        .enumerate_pixels_mut()
        .map(|(x, y, pixel)| {
            let result = camera_rays.binary_search_by_key(&(x, y), |(x, y, _)| (*x, *y));
            let rays = camera_rays[result.unwrap()].2;
            (pixel, rays)
        })
        .collect();

    let time_start = Instant::now();

    let frames = 100;
    for _ in 0..frames {
        rt_jobs.par_iter_mut().for_each(|(pixel, rays)| {
            let mut color = Vec3::zero();
            for r in rays {
                let (hit, hit_obj) = bvh.trace(&cam_pos, r);

                if let Some(obj) = hit_obj {
                    let hit_pos = cam_pos + *r * hit;
                    let light_dir = (light_pos - hit_pos).normalized();
                    let normal = (hit_pos - obj.center).normalized();
                    let ndl = light_dir.dot(normal);
                    color += obj.color * ndl / 4.;
                }
            }

            // let (closest_hit, closest_obj) = scene.tracex4(&cam_posx4, rays, false);

            // if closest_hit.cmp_lt(f32::MAX).none() {
            //     return;
            // }

            // let centers = Vec3x4::from([
            //     if let Some(o) = closest_obj[0] {
            //         o.center
            //     } else {
            //         Vec3::zero()
            //     },
            //     if let Some(o) = closest_obj[1] {
            //         o.center
            //     } else {
            //         Vec3::zero()
            //     },
            //     if let Some(o) = closest_obj[2] {
            //         o.center
            //     } else {
            //         Vec3::zero()
            //     },
            //     if let Some(o) = closest_obj[3] {
            //         o.center
            //     } else {
            //         Vec3::zero()
            //     },
            // ]);

            // let hit_pos = cam_posx4 + *rays * closest_hit;

            // let shadow_ray = (light_posx4 - hit_pos).normalized();
            // let (shadow_hit, _) = scene.tracex4(&hit_pos, &shadow_ray, false);

            // if shadow_hit.cmp_lt(f32::MAX).all() {
            //     return;
            // }

            // let shadow_hit: [f32; 4] = shadow_hit.into();

            // let light_dir = (light_posx4 - hit_pos).normalized();
            // let normal = (hit_pos - centers).normalized();
            // let ndl = light_dir.dot(normal);
            // let ndl: [f32; 4] = ndl.into();
            // let mut color = Vec3::zero();
            // for i in 0..4 {
            //     if let Some(o) = closest_obj[i] {
            //         if shadow_hit[i] >= f32::MAX {
            //             color += o.color * ndl[i] / 4.;
            //         }
            //     }
            // }

            **pixel = color_vec_to_rgb(color);
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
