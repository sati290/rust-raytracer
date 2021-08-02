extern crate image;
extern crate nalgebra as na;

use std::time::Instant;

type Vector2 = na::Vector2<f32>;
type Vector3 = na::Vector3<f32>;

struct Ray {
    origin: Vector3,
    direction: na::Unit<Vector3>,
}

impl Ray {
    fn new(origin: Vector3, direction: na::Unit<Vector3>) -> Ray {
        Ray { origin, direction }
    }
}

struct Sphere {
    center: Vector3,
    radius: f32,
    color: Vector3,
}

impl Sphere {
    fn intersect(&self, r: &Ray) -> Option<f32> {
        let r2 = self.radius * self.radius;
        let l = self.center - r.origin;
        let tca = l.dot(&r.direction);
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

fn main() {
    let subpixels = [
        Vector2::new(5. / 8., 1. / 8.),
        Vector2::new(1. / 8., 3. / 8.),
        Vector2::new(7. / 8., 5. / 8.),
        Vector2::new(3. / 8., 7. / 8.),
    ];

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

    let fov_deg: f32 = 90.;
    let plane_dist = (fov_deg.to_radians() / 2.).tan();
    let aspect = image_height as f32 / image_width as f32;
    let image_dims = Vector2::new(image_width as f32, image_height as f32);
    println!(
        "{} pixels, {} samples",
        image_width * image_height,
        image_width * image_height * subpixels.len() as u32
    );

    let time_start = Instant::now();

    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let mut color = Vector3::zeros();
        let px = Vector2::new(x as f32, (image_height - y) as f32);
        for sp in subpixels {
            let px = (px + sp).component_div(&image_dims) - Vector2::new(0.5, 0.5);
            let px = Vector3::new(px.x, px.y * aspect, plane_dist);
            let ray = Ray::new(cam_pos, na::Unit::<Vector3>::new_normalize(px));

            let mut closest_hit: Option<(&Sphere, f32)> = None;
            for o in &scene {
                if let Some(hit) = o.intersect(&ray) {
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
                let hit_pos = ray.origin + ray.direction.into_inner() * hit.1;
                let normal = na::Unit::<Vector3>::new_normalize(hit_pos - hit.0.center);
                let light_dir = na::Unit::<Vector3>::new_normalize(light_pos - hit_pos);
                let ndl = light_dir.dot(&normal);

                color += hit.0.color * ndl / subpixels.len() as f32;
            }
        }

        *pixel = color_vec_to_rgb(color);
    }

    println!("Elapsed time: {:.2?}", time_start.elapsed());

    image.save("output.png").unwrap();
}
