use nalgebra::{Scalar, SimdRealField, SimdValue, Vector3};
use rand::{Rng, distributions::Standard, prelude::Distribution};

use crate::math::Vec3f;

#[derive(Clone, Copy)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone)]
pub struct Camera {
    eye_pos: Vec3f,
    _target_pos: Vec3f,
    cam_up: Vec3f,
    horiz_fog_deg: f32,
    cam_forward: Vec3f,
    cam_right: Vec3f,
}

impl Camera {
    pub fn new(eye_pos: Vec3f, target_pos: Vec3f, cam_up: Vec3f, horiz_fog_deg: f32) -> Self {
        let cam_forward = (target_pos - eye_pos).normalize();
        let cam_right = cam_forward.cross(&cam_up).normalize();
        let cam_up = cam_right.cross(&cam_forward);
        Self {
            eye_pos,
            _target_pos: target_pos,
            cam_up,
            horiz_fog_deg,
            cam_forward,
            cam_right,
        }
    }

    pub fn position(&self) -> Vec3f {
        self.eye_pos
    }
}

pub struct CameraRayGenerator<'a> {
    camera: &'a Camera,
    region: Rect,
    pixel_size: f32,
    region_top_left: Vec3f,
    next_pixel_x: Vec3f,
    next_pixel_y: Vec3f,
    current_ray: Vec3f,
    current_x: u32,
    current_y: u32,
}

impl<'a> CameraRayGenerator<'a> {
    pub fn new(
        camera: &'a Camera,
        viewport_width: u32,
        viewport_height: u32,
        region: Rect,
    ) -> Self {
        let vp_half_width = (camera.horiz_fog_deg.to_radians() / 2.).tan();
        let vp_half_height =
            vp_half_width * ((viewport_height as f32 - 1.) / (viewport_width as f32 - 1.));

        let next_pixel_x = (2. * vp_half_width / (viewport_width as f32 - 1.)) * camera.cam_right;
        let next_pixel_y = (2. * vp_half_height / (viewport_height as f32 - 1.)) * -camera.cam_up;

        let pixel_size = vp_half_width * 2. / (viewport_width as f32 - 1.);

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

    pub fn sample1(&self, rng: &mut impl Rng) -> Vec3f {
        let u: [f32; 2] = rng.r#gen();
        let sp_offset = self.pixel_size
            * ((u[0] - 0.5) * self.camera.cam_right + (u[1] - 0.5) * self.camera.cam_up);
        (self.current_ray + sp_offset).normalize()
    }

    pub fn sample<T>(&self, rng: &mut impl Rng) -> Vector3<T>
    where
        T: Scalar + SimdRealField<Element = f32>,
        Standard: Distribution<T>,
    {
        let sp_x = rng.r#gen::<T>();
        let sp_y = rng.r#gen::<T>();

        let cam_right = Vector3::<T>::splat(self.camera.cam_right);
        let cam_up = Vector3::<T>::splat(self.camera.cam_up);
        let sp_offset = (cam_right * (sp_x - T::splat(0.5)) + cam_up * (sp_y - T::splat(0.5)))
            * T::splat(self.pixel_size);
        (Vector3::<T>::splat(self.current_ray) + sp_offset).normalize()
    }
}
