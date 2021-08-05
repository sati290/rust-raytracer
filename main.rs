use rayon::prelude::*;
use std::time::Instant;
use ultraviolet::{f32x4, Vec2, Vec3, Vec3x4};
use wide::{CmpGt, CmpLt};

struct Sphere {
    center: Vec3,
    radius: f32,
    color: Vec3,
}

impl Sphere {
    fn intersect(&self, ray_origin: &Vec3, ray_direction: &Vec3) -> Option<f32> {
        let r2 = self.radius * self.radius;
        let l = self.center - *ray_origin;
        let tca = l.dot(*ray_direction);
        let d2 = l.dot(l) - tca * tca;

        if d2 > r2 {
            None
        } else {
            let thc = (r2 - d2).sqrt();
            let t0 = tca - thc;
            let t1 = tca + thc;

            assert!(t0 <= t1);

            if t0 > 0. {
                Some(t0)
            } else if t1 > 0. {
                Some(t1)
            } else {
                None
            }
        }
    }

    fn intersectx4(&self, ray_origin: &Vec3x4, ray_direction: &Vec3x4) -> f32x4 {
        let r2 = self.radius * self.radius;
        let l = Vec3x4::splat(self.center) - *ray_origin;
        let tca = l.dot(*ray_direction);
        let d2 = l.dot(l) - tca * tca;

        let sqrt_valid = d2.cmp_lt(r2);
        let thc = (r2 - d2).sqrt();
        let t0 = tca - thc;
        let t1 = tca + thc;

        let t0_valid = t0.cmp_gt(0.) & sqrt_valid;
        let t1_valid = t1.cmp_gt(0.) & sqrt_valid;

        let t = t1_valid.blend(t1, f32x4::splat(f32::MAX));
        let t = t0_valid.blend(t1, t);

        t
    }
}

fn color_vec_to_rgb(v: Vec3) -> image::Rgb<u8> {
    image::Rgb([(v.x * 255.) as u8, (v.y * 255.) as u8, (v.z * 255.) as u8])
}

fn generate_camera_rays(
    image_width: u32,
    image_height: u32,
    horiz_fog_deg: f32,
) -> Vec<(u32, u32, Vec<Vec3>)> {
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
        Vec::<(u32, u32, Vec<Vec3>)>::with_capacity((image_width * image_height) as usize);
    for y in 0..image_height {
        for x in 0..image_width {
            let mut sp_rays = Vec::<Vec3>::with_capacity(subpixels.len());

            let px = Vec2::new(x as f32, (image_height - y) as f32);
            let px = px - Vec2::new(image_width as f32 / 2., image_height as f32 / 2.);
            for sp in subpixels {
                let px = (px + sp) * image_dims_recip;
                let ray = Vec3::new(px.x, px.y, plane_dist).normalized();
                sp_rays.push(ray);
            }

            rays.push((x, y, sp_rays));
        }
    }

    rays
}

fn main() {
    let scene = [
        Sphere {
            center: Vec3::new(0., 0., 0.),
            radius: 5.,
            color: Vec3::new(0.8, 0.8, 0.8),
        },
        Sphere {
            center: Vec3::new(5., 0., 0.),
            radius: 3.,
            color: Vec3::new(0.1, 0.8, 0.1),
        },
    ];
    let cam_pos = Vec3::new(0., 0., -20.);
    let cam_posx4 = Vec3x4::splat(cam_pos);
    let light_pos = Vec3::new(10., 10., -20.);
    let image_width = 1920;
    let image_height = 1080;
    let mut image = image::RgbImage::new(image_width, image_height);

    let mut camera_rays = generate_camera_rays(image_width, image_height, 90.);
    camera_rays.sort_by_key(|(x, y, _)| (*x, *y));

    let mut rt_jobs: Vec<_> = image
        .enumerate_pixels_mut()
        .map(|(x, y, pixel)| {
            let result = camera_rays.binary_search_by_key(&(x, y), |(x, y, _)| (*x, *y));
            let rays = camera_rays[result.unwrap()].2.clone();
            (pixel, rays)
        })
        .collect();

    let time_start = Instant::now();

    let frames = 50;
    for _ in 0..frames {
        rt_jobs.par_iter_mut().for_each(|(pixel, rays)| {
            let mut color = Vec3::zero();

            let r = Vec3x4::from([
                rays[0],
                rays[1],
                rays[2],
                rays[3]
            ]);

            let mut closest_hit = f32x4::splat(f32::MAX);
            let mut closest_obj: [Option<&Sphere>; 4] = [None; 4];
            for o in &scene {
                let hit = o.intersectx4(&cam_posx4, &r);
                let closest = hit.cmp_lt(closest_hit);
                closest_hit = closest.blend(hit, closest_hit);
                for i in 0..4 {
                    if closest[i] > 0 {
                        closest_obj[i] = Some(o);
                    }
                }
            }

            for r in &*rays {
                let mut closest_hit: Option<(&Sphere, f32)> = None;
                for o in &scene {
                    if let Some(hit) = o.intersect(&cam_pos, &r) {
                        let closest = match closest_hit {
                            Some(c) => hit < c.1,
                            None => true,
                        };

                        if closest {
                            closest_hit = Some((o, hit));
                        }
                    }
                }

                if let Some(hit) = closest_hit {
                    let hit_pos = cam_pos + *r * hit.1;
                    let normal = (hit_pos - hit.0.center).normalized();
                    let light_dir = (light_pos - hit_pos).normalized();
                    let ndl = light_dir.dot(normal);

                    color += hit.0.color * ndl / rays.len() as f32;
                }
            }

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
