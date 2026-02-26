use rand::{Rng, RngExt};
use ultraviolet::{Vec3, Vec3x4};
use wide::f32x4;

use crate::{PathInfo, Ray};

pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32
}

pub struct Camera {
    eye_pos: Vec3,
    _target_pos: Vec3,
    cam_up: Vec3,
    horiz_fog_deg: f32,
    cam_forward: Vec3,
    cam_right: Vec3,
    viewport_width: u32,
    viewport_height: u32
}

impl Camera {
    pub fn new(eye_pos: Vec3, target_pos: Vec3, cam_up: Vec3, horiz_fog_deg: f32, viewport_width: u32, viewport_height: u32) -> Self {
        let cam_forward = (target_pos - eye_pos).normalized();
        let cam_right = cam_up.cross(cam_forward).normalized();
        let cam_up = cam_forward.cross(cam_right);
        Self {
            eye_pos,
            _target_pos: target_pos,
            cam_up,
            horiz_fog_deg,
            cam_forward,
            cam_right,
            viewport_width,
            viewport_height
        }
    }

    pub fn _position(&self) -> Vec3 {
        self.eye_pos
    }

    // Generate 4 subpixel rays for each pixel
    pub fn generate_rays_4sp<R: Rng>(&self, region: &Rect, rng: &mut R, rays: &mut Vec<Ray>, path_infos: &mut Vec<PathInfo>) {
        let num_rays = (region.width * region.height * 4) as usize;
        rays.clear();
        rays.reserve(num_rays);
        path_infos.clear();
        path_infos.reserve(num_rays);

        let vp_half_width = (self.horiz_fog_deg.to_radians() / 2.).tan();
        let vp_half_height = vp_half_width * ((self.viewport_height as f32 - 1.) / (self.viewport_width as f32 - 1.));

        let next_pixel_x = (2. * vp_half_width / (self.viewport_width as f32 - 1.)) * self.cam_right;
        let next_pixel_y = (2. * vp_half_height / (self.viewport_height as f32 - 1.)) * -self.cam_up;

        let pixel_size = vp_half_width * 2. / (self.viewport_width as f32 - 1.);

        let pixel_size_x4 = f32x4::splat(pixel_size);
        let cam_right_x4 = Vec3x4::splat(self.cam_right);
        let cam_up_x4 = Vec3x4::splat(self.cam_up);

        let ray_top_left = self.cam_forward - vp_half_width * self.cam_right + vp_half_height * self.cam_up;
        let mut ray_row_start = ray_top_left + region.x as f32 * next_pixel_x + region.y as f32 * next_pixel_y;
        for ry in 0..region.height {
            let mut current_ray = ray_row_start;
            for rx in 0..region.width {
                let sp_x = f32x4::from(rng.random::<[f32;4]>());
                let sp_y = f32x4::from(rng.random::<[f32;4]>());
                let sp_offsets = pixel_size_x4 * ((sp_x - f32x4::HALF) * cam_right_x4 + (sp_y - f32x4::HALF) * cam_up_x4);
                let subpixel_rays = (Vec3x4::splat(current_ray) + sp_offsets).normalized();
                let subpixel_rays: [Vec3; 4] = subpixel_rays.into();
                for ray in &subpixel_rays {
                    rays.push(Ray::new(&self.eye_pos, ray));
                    path_infos.push(PathInfo { contribution: Vec3::one(), destination_idx: ry * region.width + rx })
                }
                current_ray += next_pixel_x;
            }
            ray_row_start += next_pixel_y;
        }
    }
}
