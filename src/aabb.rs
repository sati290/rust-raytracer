use crate::CmpGe;
use crate::Vec3;
use crate::Vec3x4;
use crate::f32x4;

#[derive(Clone, Copy)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
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
            min: Vec3::broadcast(f32::INFINITY),
            max: Vec3::broadcast(f32::NEG_INFINITY),
        }
    }

    #[must_use]
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    #[must_use]
    pub fn surface_area(&self) -> f32 {
        let size = self.size();
        2. * (size.x * size.y + size.x * size.z + size.y * size.z)
    }

    #[must_use]
    pub fn _grow(&self, point: &Vec3) -> Aabb {
        Aabb {
            min: self.min.min_by_component(*point),
            max: self.max.max_by_component(*point),
        }
    }

    pub fn grow_mut(&mut self, point: &Vec3) {
        self.min = self.min.min_by_component(*point);
        self.max = self.max.max_by_component(*point);
    }

    #[must_use]
    pub fn join(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: self.min.min_by_component(other.min),
            max: self.max.max_by_component(other.max),
        }
    }

    pub fn join_mut(&mut self, other: &Aabb) {
        self.min = self.min.min_by_component(other.min);
        self.max = self.max.max_by_component(other.max);
    }

    #[must_use]
    pub fn _intersect(&self, ray_origin: &Vec3, ray_direction_recip: &Vec3) -> bool {
        let bb_x = [self.min.x, self.max.x];
        let bb_y = [self.min.y, self.max.y];
        let bb_z = [self.min.z, self.max.z];
        let origin_dir_recip = *ray_origin * *ray_direction_recip;

        let sign_x = ray_direction_recip.x.is_sign_positive();
        let sign_y = ray_direction_recip.y.is_sign_positive();
        let sign_z = ray_direction_recip.z.is_sign_positive();
        let bb_min = Vec3::new(
            bb_x[!sign_x as usize],
            bb_y[!sign_y as usize],
            bb_z[!sign_z as usize],
        );
        let bb_max = Vec3::new(
            bb_x[sign_x as usize],
            bb_y[sign_y as usize],
            bb_z[sign_z as usize],
        );

        let tmin = bb_min.mul_add(*ray_direction_recip, -origin_dir_recip);
        let tmax = bb_max.mul_add(*ray_direction_recip, -origin_dir_recip);

        let tmin = tmin.component_max();
        let tmax = tmax.component_min();

        tmax >= tmin.max(0.)
    }

    #[must_use]
    pub fn _intersect_simd(&self, ray_origin: &Vec3x4, ray_direction_recip: &Vec3x4) -> f32x4 {
        let origin_dir_recip = *ray_origin * *ray_direction_recip;

        let tx1 = f32x4::splat(self.min.x).mul_sub(ray_direction_recip.x, origin_dir_recip.x);
        let tx2 = f32x4::splat(self.max.x).mul_sub(ray_direction_recip.x, origin_dir_recip.x);

        let tmin = tx1.min(tx2);
        let tmax = tx1.max(tx2);

        let ty1 = f32x4::splat(self.min.y).mul_sub(ray_direction_recip.y, origin_dir_recip.y);
        let ty2 = f32x4::splat(self.max.y).mul_sub(ray_direction_recip.y, origin_dir_recip.y);

        let tmin = tmin.max(ty1.min(ty2));
        let tmax = tmax.min(ty1.max(ty2));

        let tz1 = f32x4::splat(self.min.z).mul_sub(ray_direction_recip.z, origin_dir_recip.z);
        let tz2 = f32x4::splat(self.max.z).mul_sub(ray_direction_recip.z, origin_dir_recip.z);

        let tmin = tmin.max(tz1.min(tz2));
        let tmax = tmax.min(tz1.max(tz2));

        tmax.cmp_ge(tmin.max(f32x4::ZERO))
    }
}
