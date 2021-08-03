extern crate image;
extern crate nalgebra as na;

use std::time::Instant;

type Vector2 = na::Vector2<f32>;
type Vector3 = na::Vector3<f32>;
type UVector3 = na::Unit<Vector3>;
type Matrix4 = na::Matrix4<f32>;

struct Sphere {
    center: Vector3,
    radius: f32,
    color: Vector3,
}

impl Sphere {
    fn intersect(&self, ray_origin: &Vector3, ray_direction: &UVector3) -> Option<f32> {
        let r2 = self.radius * self.radius;
        let l = self.center - ray_origin;
        let tca = l.dot(ray_direction);
        let d2 = l.dot(&l) - tca * tca;

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
}

fn color_vec_to_rgb(v: Vector3) -> image::Rgb<u8> {
    image::Rgb([(v.x * 255.) as u8, (v.y * 255.) as u8, (v.z * 255.) as u8])
}

fn generate_camera_rays(
    image_width: u32,
    image_height: u32,
    horiz_fog_deg: f32,
) -> Vec<(u32, u32, Vec<UVector3>)> {
    let subpixels = [
        Vector2::new(5. / 8., 1. / 8.),
        Vector2::new(1. / 8., 3. / 8.),
        Vector2::new(7. / 8., 5. / 8.),
        Vector2::new(3. / 8., 7. / 8.),
    ];

    let plane_dist = (horiz_fog_deg.to_radians() / 2.).tan();
    let image_dims_recip = Vector2::new(
        1. / image_width as f32,
        1. / image_width /* width used here to handle aspect */ as f32,
    );

    let mut rays =
        Vec::<(u32, u32, Vec<UVector3>)>::with_capacity((image_width * image_height) as usize);
    for x in 0..image_width {
        for y in 0..image_height {
            let mut sp_rays = Vec::<UVector3>::with_capacity(subpixels.len());

            let px = Vector2::new(x as f32, (image_height - y) as f32);
            let px = px - Vector2::new(image_width as f32 / 2., image_height as f32 / 2.);
            for sp in subpixels {
                let px = (px + sp).component_mul(&image_dims_recip);
                let ray = UVector3::new_normalize(Vector3::new(px.x, px.y, plane_dist));
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
            center: Vector3::new(0., 0., 0.),
            radius: 5.,
            color: Vector3::new(0.8, 0.8, 0.8),
        },
        Sphere {
            center: Vector3::new(5., 0., 0.),
            radius: 3.,
            color: Vector3::new(0.1, 0.8, 0.1),
        },
    ];
    let cam_pos = Vector3::new(0., 0., -20.);
    let light_pos = Vector3::new(10., 10., -20.);
    let image_width = 1920;
    let image_height = 1080;
    let mut image = image::RgbImage::new(image_width, image_height);

    let camera_rays = generate_camera_rays(image_width, image_height, 90.);

    let time_start = Instant::now();

    let frames = 50;
    for _ in 0..frames {
        for (x, y, rays) in &camera_rays {
            let mut color = Vector3::zeros();
            for r in rays {
                let mut closest_hit: Option<(&Sphere, f32)> = None;
                for o in &scene {
                    if let Some(hit) = o.intersect(&cam_pos, r) {
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
                    let hit_pos = cam_pos + r.into_inner() * hit.1;
                    let normal = UVector3::new_normalize(hit_pos - hit.0.center);
                    let light_dir = UVector3::new_normalize(light_pos - hit_pos);
                    let ndl = light_dir.dot(&normal);

                    color += hit.0.color * ndl / rays.len() as f32;
                }
            }

            image.put_pixel(*x, *y, color_vec_to_rgb(color));
        }
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
