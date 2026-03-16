use ultraviolet::{Vec3, Vec3x4, Vec4};
use wide::f32x4;

#[derive(Clone)]
#[repr(C, align(16))]
pub struct StreamRay {
    pub origin_far: Vec4,           // x, y, z, near
    pub direction_recip_near: Vec4, // x, y, z, far
    pub direction: Vec4,
}

impl StreamRay {
    #[must_use]
    pub fn new(origin: &Vec3, direction: &Vec3, near: f32, far: f32) -> Self {
        let dir_recip = Vec3::one() / *direction;
        StreamRay {
            origin_far: Vec4::new(origin.x, origin.y, origin.z, far),
            direction: Vec4::from(*direction),
            direction_recip_near: Vec4::new(dir_recip.x, dir_recip.y, dir_recip.z, near),
        }
    }

    #[must_use]
    pub fn _is_hit(&self) -> bool {
        self.origin_far.w < f32::INFINITY
    }

    #[must_use]
    pub fn hit_dist(&self) -> f32 {
        self.origin_far.w
    }

    #[must_use]
    pub fn hit_pos(&self) -> Vec3 {
        self.origin_far.xyz() + self.direction.xyz() * self.hit_dist()
    }
}

#[derive(Clone)]
pub struct Ray4 {
    pub origin: Vec3x4,
    pub direction: Vec3x4,
    pub near: f32x4,
    pub far: f32x4,
    pub valid: f32x4,
}

impl Ray4 {
    #[must_use]
    pub fn new(
        origin: &Vec3x4,
        direction: &Vec3x4,
        near: &f32x4,
        far: &f32x4,
        valid: &f32x4,
    ) -> Self {
        Ray4 {
            origin: *origin,
            direction: *direction,
            near: valid.blend(*near, f32x4::splat(f32::INFINITY)),
            far: valid.blend(*far, f32x4::splat(f32::NEG_INFINITY)),
            valid: *valid,
        }
    }

    #[must_use]
    pub fn hit_dist(&self) -> f32x4 {
        self.far
    }

    #[must_use]
    pub fn hit_pos(&self) -> Vec3x4 {
        self.origin + self.direction * self.hit_dist()
    }
}

#[derive(Clone)]
pub struct RayHit4 {
    pub ray: Ray4,
    pub obj_idx: [Option<u32>; 4],
}

impl From<Ray4> for RayHit4 {
    fn from(ray: Ray4) -> Self {
        RayHit4 {
            ray,
            obj_idx: [None; _],
        }
    }
}
