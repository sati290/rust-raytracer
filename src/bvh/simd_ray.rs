use safe_arch::*;

use crate::ray::{Ray4, Ray8};

macro_rules! simd_ray_n {
    ($(($name:ident, $n:literal, $t:ident, $ray:ident, $splat:ident)),+) => {
        $(
            pub struct $name {
                pub dir_recip_x: $t,
                pub dir_recip_y: $t,
                pub dir_recip_z: $t,
                pub origin_dir_recip_x: $t,
                pub origin_dir_recip_y: $t,
                pub origin_dir_recip_z: $t,
                pub near: $t,
                pub far: $t,
            }

            impl $name {
                #[inline]
                pub fn update_far(&mut self, far: &[f32; $n]) {
                    self.far = $t::from(*far);
                }
            }

            impl From<&$ray> for $name {
                #[inline]
                fn from(ray: &$ray) -> Self {
                    use safe_arch::*;

                    let origin_x = $t::from(ray.origin.x.to_array());
                    let origin_y = $t::from(ray.origin.y.to_array());
                    let origin_z = $t::from(ray.origin.z.to_array());
                    let near = $t::from(ray.near.to_array());
                    let far = $t::from(ray.far.to_array());

                    let dir_x = $t::from(ray.direction.x.to_array());
                    let dir_y = $t::from(ray.direction.y.to_array());
                    let dir_z = $t::from(ray.direction.z.to_array());

                    let one = $splat(1.);
                    let dir_recip_x = one / dir_x;
                    let dir_recip_y = one / dir_y;
                    let dir_recip_z = one / dir_z;

                    let origin_dir_recip_x = origin_x * dir_recip_x;
                    let origin_dir_recip_y = origin_y * dir_recip_y;
                    let origin_dir_recip_z = origin_z * dir_recip_z;

                    $name {
                        dir_recip_x,
                        dir_recip_y,
                        dir_recip_z,
                        origin_dir_recip_x,
                        origin_dir_recip_y,
                        origin_dir_recip_z,
                        near,
                        far,
                    }
                }
            }
        )+
    }
}

simd_ray_n!(
    (SimdRay4, 4, m128, Ray4, set_splat_m128),
    (SimdRay8, 8, m256, Ray8, set_splat_m256)
);

// Layout
// R1   R1   R2   R2   R3   R3   R4   R4
pub struct SimdRay4x2Interleaved {
    pub dir_recip_x: m256,
    pub dir_recip_y: m256,
    pub dir_recip_z: m256,
    pub origin_dir_recip_x: m256,
    pub origin_dir_recip_y: m256,
    pub origin_dir_recip_z: m256,
    pub near: m256,
    pub far: m256,
}

impl SimdRay4x2Interleaved {
    #[inline]
    pub fn update_far(&mut self, far: &[f32; 4]) {
        self.far = set_m256(
            far[3], far[3], far[2], far[2], far[1], far[1], far[0], far[0],
        );
    }
}

impl From<&Ray4> for SimdRay4x2Interleaved {
    #[inline]
    fn from(ray: &Ray4) -> Self {
        use safe_arch::*;

        let origin_x = {
            let a = ray.origin.x.as_array_ref();
            set_m256(a[3], a[3], a[2], a[2], a[1], a[1], a[0], a[0])
        };
        let origin_y = {
            let a = ray.origin.y.as_array_ref();
            set_m256(a[3], a[3], a[2], a[2], a[1], a[1], a[0], a[0])
        };
        let origin_z = {
            let a = ray.origin.z.as_array_ref();
            set_m256(a[3], a[3], a[2], a[2], a[1], a[1], a[0], a[0])
        };
        let near = {
            let a = ray.near.as_array_ref();
            set_m256(a[3], a[3], a[2], a[2], a[1], a[1], a[0], a[0])
        };
        let far = {
            let a = ray.far.as_array_ref();
            set_m256(a[3], a[3], a[2], a[2], a[1], a[1], a[0], a[0])
        };

        let dir_x = {
            let a = ray.direction.x.as_array_ref();
            set_m256(a[3], a[3], a[2], a[2], a[1], a[1], a[0], a[0])
        };
        let dir_y = {
            let a = ray.direction.y.as_array_ref();
            set_m256(a[3], a[3], a[2], a[2], a[1], a[1], a[0], a[0])
        };
        let dir_z = {
            let a = ray.direction.z.as_array_ref();
            set_m256(a[3], a[3], a[2], a[2], a[1], a[1], a[0], a[0])
        };

        let one = set_splat_m256(1.);
        let dir_recip_x = one / dir_x;
        let dir_recip_y = one / dir_y;
        let dir_recip_z = one / dir_z;

        let origin_dir_recip_x = origin_x * dir_recip_x;
        let origin_dir_recip_y = origin_y * dir_recip_y;
        let origin_dir_recip_z = origin_z * dir_recip_z;

        SimdRay4x2Interleaved {
            dir_recip_x,
            dir_recip_y,
            dir_recip_z,
            origin_dir_recip_x,
            origin_dir_recip_y,
            origin_dir_recip_z,
            near,
            far,
        }
    }
}
