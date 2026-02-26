use ultraviolet::{Vec3, Vec4};

#[repr(C, align(16))]
pub struct Ray {
    pub origin_near: Vec4,         // x, y, z, near
    pub direction_recip_far: Vec4, // x, y, z, far
    pub direction: Vec4,
}

impl Ray {
    #[must_use]
    pub fn new(origin: &Vec3, direction: &Vec3, near: f32) -> Self {
        let dir_recip = Vec3::one() / *direction;
        Ray {
            origin_near: Vec4::new(origin.x, origin.y, origin.z, near),
            direction: Vec4::from(*direction),
            direction_recip_far: Vec4::new(dir_recip.x, dir_recip.y, dir_recip.z, f32::INFINITY),
        }
    }

    #[must_use]
    pub fn _is_hit(&self) -> bool {
        self.direction_recip_far.w < f32::INFINITY
    }

    #[must_use]
    pub fn hit_dist(&self) -> f32 {
        self.direction_recip_far.w
    }

    #[must_use]
    pub fn hit_pos(&self) -> Vec3 {
        self.origin_near.xyz() + self.direction.xyz() * self.hit_dist()
    }
}
