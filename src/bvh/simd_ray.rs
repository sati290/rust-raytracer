use safe_arch::*;
use simba::simd::{WideF32x4, WideF32x8};

use crate::ray::Ray;

pub trait SimdRay<T>: for<'a> From<&'a Ray<T>> {
    fn update_far(&mut self, far: &T);
}

macro_rules! simd_ray_n {
    ($(($name:ident, $n:literal, $st:ident, $t:ty, $splat:ident)),+) => {
        $(
            pub struct $name {
                pub dir_recip_x: $st,
                pub dir_recip_y: $st,
                pub dir_recip_z: $st,
                pub origin_dir_recip_x: $st,
                pub origin_dir_recip_y: $st,
                pub origin_dir_recip_z: $st,
                pub near: $st,
                pub far: $st,
            }

            impl SimdRay<$t> for $name {
                #[inline]
                fn update_far(&mut self, far: &$t) {
                    self.far = $st::from(far.into_arr());
                }
            }

            impl From<&Ray<$t>> for $name {
                #[inline]
                fn from(ray: &Ray<$t>) -> Self {
                    use safe_arch::*;

                    let origin_x = $st::from_array(ray.origin.x.into());
                    let origin_y = $st::from_array(ray.origin.y.into());
                    let origin_z = $st::from_array(ray.origin.z.into());
                    let near = $st::from_array(ray.near.into());
                    let far = $st::from_array(ray.far.into());

                    let dir_x = $st::from_array(ray.direction.x.into());
                    let dir_y = $st::from_array(ray.direction.y.into());
                    let dir_z = $st::from_array(ray.direction.z.into());

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
    (SimdRay4, 4, m128, WideF32x4, set_splat_m128),
    (SimdRay8, 8, m256, WideF32x8, set_splat_m256)
);
