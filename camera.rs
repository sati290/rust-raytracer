use ultraviolet::{Vec2, Vec3, Vec3x4};

use crate::Ray;

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

    pub fn position(&self) -> Vec3 {
        self.eye_pos
    }

    pub fn generate_rays(&self, region: &Rect) -> Vec<Ray> {
        const SUBPIXELS: [Vec2; 4] = [
            Vec2::new(1. / 8., -3. / 8.),
            Vec2::new(-3. / 8., -1. / 8.),
            Vec2::new(3. / 8., 1. / 8.),
            Vec2::new(-1. / 8., 3. / 8.),
        ];

        let mut rays = Vec::with_capacity(region.width as usize * region.height as usize * SUBPIXELS.len());

        let vp_half_width = (self.horiz_fog_deg.to_radians() / 2.).tan();
        let vp_half_height = vp_half_width * ((self.viewport_height as f32 - 1.) / (self.viewport_width as f32 - 1.));

        let next_pixel_x = (2. * vp_half_width / (self.viewport_width as f32 - 1.)) * self.cam_right;
        let next_pixel_y = (2. * vp_half_height / (self.viewport_height as f32 - 1.)) * -self.cam_up;

        let pixel_size = vp_half_width * 2. / (self.viewport_width as f32 - 1.);
        let subpixel_offsets = Vec3x4::from(SUBPIXELS.map(|sp| pixel_size * (sp.x * self.cam_right + sp.y * self.cam_up)));

        let ray_top_left = self.cam_forward - vp_half_width * self.cam_right + vp_half_height * self.cam_up;
        for y in region.y..region.y + region.height {
            let mut current_ray = ray_top_left + region.x as f32 * next_pixel_x + y as f32 * next_pixel_y;
            for _ in region.x..region.x + region.width {
                let subpixel_rays = (Vec3x4::splat(current_ray) + subpixel_offsets).normalized();
                let subpixel_rays: [Vec3; 4] = subpixel_rays.into();
                subpixel_rays.iter().for_each(|ray| {
                    rays.push(Ray::new(&self.eye_pos, ray));
                });
                current_ray += next_pixel_x;
            }
        }

        rays
    }
}