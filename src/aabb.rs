use nalgebra::SimdComplexField;
use nalgebra::SimdPartialOrd;
use nalgebra::SimdValue;
use simba::simd::WideBoolF32x4;
use simba::simd::WideF32x4;

use crate::math::Vec3f;
use crate::math::Vec3x4f;

#[derive(Clone, Copy)]
pub struct Aabb {
    pub min: Vec3f,
    pub max: Vec3f,
}

impl Default for Aabb {
    fn default() -> Self {
        Aabb::empty()
    }
}

impl Aabb {
    #[must_use]
    pub fn empty() -> Aabb {
        Aabb {
            min: Vec3f::from_element(f32::INFINITY),
            max: Vec3f::from_element(f32::NEG_INFINITY),
        }
    }

    #[must_use]
    pub fn size(&self) -> Vec3f {
        self.max - self.min
    }

    #[must_use]
    pub fn surface_area(&self) -> f32 {
        let size = self.size();
        2. * (size.x * size.y + size.x * size.z + size.y * size.z)
    }

    #[must_use]
    pub fn _grow(&self, point: &Vec3f) -> Aabb {
        Aabb {
            min: self.min.inf(point),
            max: self.max.sup(point),
        }
    }

    pub fn grow_mut(&mut self, point: &Vec3f) {
        self.min = self.min.inf(point);
        self.max = self.max.sup(point);
    }

    #[must_use]
    pub fn join(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: self.min.inf(&other.min),
            max: self.max.sup(&other.max),
        }
    }

    pub fn join_mut(&mut self, other: &Aabb) {
        self.min = self.min.inf(&other.min);
        self.max = self.max.sup(&other.max);
    }

    #[must_use]
    pub fn _intersect(&self, ray_origin: &Vec3f, ray_direction_recip: &Vec3f) -> bool {
        let bb_x = [self.min.x, self.max.x];
        let bb_y = [self.min.y, self.max.y];
        let bb_z = [self.min.z, self.max.z];
        let origin_dir_recip = ray_origin.component_mul(ray_direction_recip);

        let sign_x = ray_direction_recip.x.is_sign_positive();
        let sign_y = ray_direction_recip.y.is_sign_positive();
        let sign_z = ray_direction_recip.z.is_sign_positive();
        let bb_min = Vec3f::new(
            bb_x[!sign_x as usize],
            bb_y[!sign_y as usize],
            bb_z[!sign_z as usize],
        );
        let bb_max = Vec3f::new(
            bb_x[sign_x as usize],
            bb_y[sign_y as usize],
            bb_z[sign_z as usize],
        );

        let tmin = bb_min.zip_zip_map(ray_direction_recip, &origin_dir_recip, |a, b, c| {
            a.mul_add(b, -c)
        });
        let tmax = bb_max.zip_zip_map(ray_direction_recip, &origin_dir_recip, |a, b, c| {
            a.mul_add(b, -c)
        });

        let tmin = tmin.max();
        let tmax = tmax.min();

        tmax >= tmin.max(0.)
    }

    #[must_use]
    pub fn _intersect_simd(
        &self,
        ray_origin: &Vec3x4f,
        ray_direction_recip: &Vec3x4f,
        ray_near: &WideF32x4,
        ray_far: &WideF32x4,
    ) -> WideBoolF32x4 {
        let origin_dir_recip = ray_origin.component_mul(ray_direction_recip);

        let tx1 =
            WideF32x4::splat(self.min.x).simd_mul_add(ray_direction_recip.x, -origin_dir_recip.x);
        let tx2 =
            WideF32x4::splat(self.max.x).simd_mul_add(ray_direction_recip.x, -origin_dir_recip.x);

        let tmin = tx1.simd_min(tx2);
        let tmax = tx1.simd_max(tx2);

        let ty1 =
            WideF32x4::splat(self.min.y).simd_mul_add(ray_direction_recip.y, -origin_dir_recip.y);
        let ty2 =
            WideF32x4::splat(self.max.y).simd_mul_add(ray_direction_recip.y, -origin_dir_recip.y);

        let tmin = tmin.simd_max(ty1.simd_min(ty2));
        let tmax = tmax.simd_min(ty1.simd_max(ty2));

        let tz1 =
            WideF32x4::splat(self.min.z).simd_mul_add(ray_direction_recip.z, -origin_dir_recip.z);
        let tz2 =
            WideF32x4::splat(self.max.z).simd_mul_add(ray_direction_recip.z, -origin_dir_recip.z);

        let tmin = tmin.simd_max(tz1.simd_min(tz2));
        let tmax = tmax.simd_min(tz1.simd_max(tz2));

        let tmin = tmin.simd_max(*ray_near);
        let tmax = tmax.simd_min(*ray_far);

        tmax.simd_ge(tmin)
    }
}
