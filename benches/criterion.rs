use std::cell::Cell;

use criterion::{Criterion, criterion_group, criterion_main};
use rand::RngExt;
use ultraviolet::{Vec2, Vec3, Vec3x4, Vec4};

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

pub struct Rect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

pub struct Camera {
    _eye_pos: Vec3,
    _target_pos: Vec3,
    cam_up: Vec3,
    horiz_fog_deg: f32,
    cam_forward: Vec3,
    cam_right: Vec3,
}

impl Camera {
    pub fn new(eye_pos: Vec3, target_pos: Vec3, cam_up: Vec3, horiz_fog_deg: f32) -> Self {
        let cam_forward = (target_pos - eye_pos).normalized();
        let cam_right = cam_up.cross(cam_forward).normalized();
        let cam_up = cam_forward.cross(cam_right);
        Self {
            _eye_pos: eye_pos,
            _target_pos: target_pos,
            cam_up,
            horiz_fog_deg,
            cam_forward,
            cam_right,
        }
    }

    pub fn generate_rays(
        &self,
        viewport_width_px: u32,
        viewport_height_px: u32,
        region: &Rect,
        rays: &mut Vec<Ray>,
    ) {
        const SUBPIXELS: [Vec2; 4] = [
            Vec2::new(1. / 8., -3. / 8.),
            Vec2::new(-3. / 8., -1. / 8.),
            Vec2::new(3. / 8., 1. / 8.),
            Vec2::new(-1. / 8., 3. / 8.),
        ];

        rays.clear();
        rays.reserve(viewport_width_px as usize * viewport_height_px as usize * SUBPIXELS.len());

        let vp_half_width = (self.horiz_fog_deg.to_radians() / 2.).tan();
        let vp_half_height =
            vp_half_width * ((viewport_height_px as f32 - 1.) / (viewport_width_px as f32 - 1.));

        let next_pixel_x = (2. * vp_half_width / (viewport_width_px as f32 - 1.)) * self.cam_right;
        let next_pixel_y = (2. * vp_half_height / (viewport_height_px as f32 - 1.)) * -self.cam_up;

        let pixel_size = vp_half_width * 2. / (viewport_width_px as f32 - 1.);
        let subpixel_offsets = Vec3x4::from(
            SUBPIXELS.map(|sp| pixel_size * (sp.x * self.cam_right + sp.y * self.cam_up)),
        );

        let ray_top_left =
            self.cam_forward - vp_half_width * self.cam_right + vp_half_height * self.cam_up;
        for y in region.y..region.height {
            let mut current_ray =
                ray_top_left + region.x as f32 * next_pixel_x + y as f32 * next_pixel_y;
            for _ in region.x..region.width {
                let subpixel_rays = (Vec3x4::splat(current_ray) + subpixel_offsets).normalized();
                let subpixel_rays: [Vec3; 4] = subpixel_rays.into();
                subpixel_rays.iter().for_each(|ray| {
                    rays.push(Ray::new(&self._eye_pos, ray));
                });
                current_ray += next_pixel_x;
            }
        }
    }
}

fn camera_benchmark(c: &mut Criterion) {
    let camera = Camera::new(
        Vec3::new(0., 0., 0.),
        Vec3::new(0., 0., 1.),
        Vec3::new(0., 1., 0.),
        90.,
    );
    let region = Rect {
        x: 0,
        y: 0,
        width: 1920,
        height: 1080,
    };
    let region2 = Rect {
        x: 0,
        y: 0,
        width: 32,
        height: 32,
    };

    let mut rays: Vec<Ray> = Vec::new();
    c.bench_function("generate_camera_rays_v2", |b| {
        b.iter(|| camera.generate_rays(1920, 1080, &region, &mut rays))
    });
    c.bench_function("generate_camera_rays_v2_32x32", |b| {
        b.iter(|| camera.generate_rays(1920, 1080, &region2, &mut rays))
    });
}

fn rand_benchmark(c: &mut Criterion) {
    let mut rng = rand::rng();

    let mut nums: [f32; _] = [0.; 8192];
    c.bench_function("rand_f32", |b| {
        b.iter(|| nums.iter_mut().for_each(|n| *n = rng.random()))
    });
}

fn cell_benchmark(c: &mut Criterion) {
    const N: usize = 8192;
    let mut vec = vec![0.; N];
    let vec_cell = vec![Cell::new(0.); N];

    c.bench_function("vec", |b| b.iter(|| vec.iter_mut().for_each(|n| *n += 1.)));
    c.bench_function("vec_cell", |b| {
        b.iter(|| vec_cell.iter().for_each(|n| n.update(|x| x + 1.)))
    });
}

criterion_group!(benches, camera_benchmark, rand_benchmark, cell_benchmark);
criterion_main!(benches);
