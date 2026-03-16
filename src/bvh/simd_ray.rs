use safe_arch::*;

use crate::ray::Ray4;

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

impl From<&Ray4> for SimdRay4x2Interleaved {
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

impl SimdRay4x2Interleaved {
    pub fn update_far(&mut self, far: &[f32; 4]) {
        self.far = set_m256(
            far[3], far[3], far[2], far[2], far[1], far[1], far[0], far[0],
        );
    }
}
