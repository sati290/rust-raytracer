use rand::{Rng, RngExt};
use ultraviolet::{Vec3, Vec3x4, Vec3x8};
use wide::{f32x4, f32x8};

use crate::{PathInfo, Ray};

#[derive(Clone, Copy)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

pub struct Camera {
    eye_pos: Vec3,
    _target_pos: Vec3,
    cam_up: Vec3,
    horiz_fog_deg: f32,
    cam_forward: Vec3,
    cam_right: Vec3,
    viewport_width: u32,
    viewport_height: u32,
}

impl Camera {
    pub fn new(
        eye_pos: Vec3,
        target_pos: Vec3,
        cam_up: Vec3,
        horiz_fog_deg: f32,
        viewport_width: u32,
        viewport_height: u32,
    ) -> Self {
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
            viewport_height,
        }
    }

    pub fn position(&self) -> Vec3 {
        self.eye_pos
    }

    // Generate 4 subpixel rays for each pixel
    pub fn _generate_rays_4sp<R: Rng>(
        &self,
        region: &Rect,
        rng: &mut R,
        rays: &mut Vec<Ray>,
        path_infos: &mut Vec<PathInfo>,
    ) {
        let num_rays = (region.width * region.height * 4) as usize;
        rays.clear();
        rays.reserve(num_rays);
        path_infos.clear();
        path_infos.reserve(num_rays);

        let vp_half_width = (self.horiz_fog_deg.to_radians() / 2.).tan();
        let vp_half_height = vp_half_width
            * ((self.viewport_height as f32 - 1.) / (self.viewport_width as f32 - 1.));

        let next_pixel_x =
            (2. * vp_half_width / (self.viewport_width as f32 - 1.)) * self.cam_right;
        let next_pixel_y =
            (2. * vp_half_height / (self.viewport_height as f32 - 1.)) * -self.cam_up;

        let pixel_size = vp_half_width * 2. / (self.viewport_width as f32 - 1.);

        let pixel_size_x4 = f32x4::splat(pixel_size);
        let cam_right_x4 = Vec3x4::splat(self.cam_right);
        let cam_up_x4 = Vec3x4::splat(self.cam_up);

        let ray_top_left =
            self.cam_forward - vp_half_width * self.cam_right + vp_half_height * self.cam_up;
        let mut ray_row_start =
            ray_top_left + region.x as f32 * next_pixel_x + region.y as f32 * next_pixel_y;
        for ry in 0..region.height {
            let mut current_ray = ray_row_start;
            for rx in 0..region.width {
                let sp_x = f32x4::from(rng.random::<[f32; 4]>());
                let sp_y = f32x4::from(rng.random::<[f32; 4]>());
                let sp_offsets = pixel_size_x4
                    * ((sp_x - f32x4::HALF) * cam_right_x4 + (sp_y - f32x4::HALF) * cam_up_x4);
                let subpixel_rays = (Vec3x4::splat(current_ray) + sp_offsets).normalized();
                let subpixel_rays: [Vec3; 4] = subpixel_rays.into();
                for ray in &subpixel_rays {
                    rays.push(Ray::new(&self.eye_pos, ray, 0., f32::INFINITY));
                    path_infos.push(PathInfo {
                        weight: Vec3::one(),
                        destination_idx: ry * region.width + rx,
                        bounces: 0,
                    })
                }
                current_ray += next_pixel_x;
            }
            ray_row_start += next_pixel_y;
        }
    }
}

pub struct CameraRayGenerator<'a> {
    camera: &'a Camera,
    region: Rect,
    pixel_size: f32,
    region_top_left: Vec3,
    next_pixel_x: Vec3,
    next_pixel_y: Vec3,
    current_ray: Vec3,
    current_x: u32,
    current_y: u32,
}

impl<'a> CameraRayGenerator<'a> {
    pub fn new(camera: &'a Camera, region: Rect) -> Self {
        let vp_half_width = (camera.horiz_fog_deg.to_radians() / 2.).tan();
        let vp_half_height = vp_half_width
            * ((camera.viewport_height as f32 - 1.) / (camera.viewport_width as f32 - 1.));

        let next_pixel_x =
            (2. * vp_half_width / (camera.viewport_width as f32 - 1.)) * camera.cam_right;
        let next_pixel_y =
            (2. * vp_half_height / (camera.viewport_height as f32 - 1.)) * -camera.cam_up;

        let pixel_size = vp_half_width * 2. / (camera.viewport_width as f32 - 1.);

        let top_left =
            camera.cam_forward - vp_half_width * camera.cam_right + vp_half_height * camera.cam_up;
        let region_top_left =
            top_left + region.x as f32 * next_pixel_x + region.y as f32 * next_pixel_y;

        CameraRayGenerator {
            camera,
            region,
            pixel_size,
            region_top_left,
            next_pixel_x,
            next_pixel_y,
            current_x: 0,
            current_y: 0,
            current_ray: region_top_left,
        }
    }

    pub fn current_pixel_idx(&self) -> u32 {
        self.current_y * self.region.width + self.current_x
    }

    pub fn is_done(&self) -> bool {
        self.current_y == self.region.height && self.current_x == self.region.width
    }

    pub fn next_pixel(&mut self) {
        if self.is_done() {
            return;
        }

        self.current_x += 1;
        if self.current_x < self.region.width {
            self.current_ray += self.next_pixel_x;
        } else {
            self.current_y += 1;
            if self.current_y < self.region.height {
                self.current_x = 0;
                self.current_ray = self.region_top_left + self.current_y as f32 * self.next_pixel_y;
            }
        }
    }

    pub fn sample<R: Rng>(&self, rng: &mut R) -> Vec3 {
        let u: [f32; 2] = rng.random();
        let sp_offset = self.pixel_size
            * ((u[0] - 0.5) * self.camera.cam_right + (u[1] - 0.5) * self.camera.cam_up);
        (self.current_ray + sp_offset).normalized()
    }

    pub fn _sample4<R: Rng>(&self, rng: &mut R) -> Vec3x4 {
        let sp_x = f32x4::from(rng.random::<[f32; 4]>());
        let sp_y = f32x4::from(rng.random::<[f32; 4]>());

        let sp_offset = f32x4::splat(self.pixel_size)
            * ((sp_x - f32x4::HALF) * Vec3x4::splat(self.camera.cam_right)
                + (sp_y - f32x4::HALF) * Vec3x4::splat(self.camera.cam_up));
        (Vec3x4::splat(self.current_ray) + sp_offset).normalized()
    }

    pub fn sample8<R: Rng>(&self, rng: &mut R) -> Vec3x8 {
        let sp_x = f32x8::from(rng.random::<[f32; 8]>());
        let sp_y = f32x8::from(rng.random::<[f32; 8]>());

        let sp_offset = f32x8::splat(self.pixel_size)
            * ((sp_x - f32x8::HALF) * Vec3x8::splat(self.camera.cam_right)
                + (sp_y - f32x8::HALF) * Vec3x8::splat(self.camera.cam_up));
        (Vec3x8::splat(self.current_ray) + sp_offset).normalized()
    }
}
