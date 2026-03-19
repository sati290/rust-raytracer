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
