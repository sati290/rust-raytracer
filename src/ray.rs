use ultraviolet::{Vec3, Vec3x4, Vec3x8, Vec4};
use wide::{f32x4, f32x8};

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

macro_rules! ray_n {
    ($(($n:ident, $nh:ident, $c:literal, $t:ident, $vt:ident)),+) => {
        $(
            #[derive(Clone)]
            pub struct $n {
                pub origin: $vt,
                pub direction: $vt,
                pub near: $t,
                pub far: $t,
                pub valid: $t,
            }

            impl $n {
                #[must_use]
                pub fn new(
                    origin: &$vt,
                    direction: &$vt,
                    near: &$t,
                    far: &$t,
                    valid: &$t,
                ) -> Self {
                    $n {
                        origin: *origin,
                        direction: *direction,
                        near: valid.blend(*near, $t::splat(f32::INFINITY)),
                        far: valid.blend(*far, $t::splat(f32::NEG_INFINITY)),
                        valid: *valid,
                    }
                }

                #[must_use]
                pub fn hit_dist(&self) -> $t {
                    self.far
                }

                #[must_use]
                pub fn hit_pos(&self) -> $vt {
                    self.origin + self.direction * self.hit_dist()
                }
            }

            #[derive(Clone)]
            pub struct $nh {
                pub ray: $n,
                pub obj_idx: [Option<u32>; $c],
            }

            impl From<$n> for $nh {
                fn from(ray: $n) -> Self {
                    $nh {
                        ray,
                        obj_idx: [None; _],
                    }
                }
            }
        )+
    }
}

ray_n!(
    (Ray4, RayHit4, 4, f32x4, Vec3x4),
    (Ray8, RayHit8, 8, f32x8, Vec3x8)
);
