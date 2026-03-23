use safe_arch::*;
use simba::simd::{WideBoolF32x8, WideF32x8};

use crate::{
    bvh::{node_intersector::BvhNodeIntersector, simd_ray::SimdRay8},
    math::Vec3x4f,
};

pub struct BvhNodeIntersector8 {}

impl BvhNodeIntersector<WideF32x8> for BvhNodeIntersector8 {
    type SimdRay = SimdRay8;

    #[inline(always)]
    fn intersect(child_bbox: &Vec3x4f, index: usize, ray: &SimdRay8) -> (WideBoolF32x8, WideF32x8) {
        // R1   R2   R3   R4   R5   R6   R7   R8
        // MINL MINR MAXL MAXR
        let bbx = child_bbox.x.into_arr();
        let bby = child_bbox.y.into_arr();
        let bbz = child_bbox.z.into_arr();

        let bb_min_x = set_splat_m256(bbx[index]);
        let bb_min_y = set_splat_m256(bby[index]);
        let bb_min_z = set_splat_m256(bbz[index]);

        let bb_max_x = set_splat_m256(bbx[index + 2]);
        let bb_max_y = set_splat_m256(bby[index + 2]);
        let bb_max_z = set_splat_m256(bbz[index + 2]);

        let tmin_x = fused_mul_sub_m256(bb_min_x, ray.dir_recip_x, ray.origin_dir_recip_x);
        let tmin_y = fused_mul_sub_m256(bb_min_y, ray.dir_recip_y, ray.origin_dir_recip_y);
        let tmin_z = fused_mul_sub_m256(bb_min_z, ray.dir_recip_z, ray.origin_dir_recip_z);
        let tmax_x = fused_mul_sub_m256(bb_max_x, ray.dir_recip_x, ray.origin_dir_recip_x);
        let tmax_y = fused_mul_sub_m256(bb_max_y, ray.dir_recip_y, ray.origin_dir_recip_y);
        let tmax_z = fused_mul_sub_m256(bb_max_z, ray.dir_recip_z, ray.origin_dir_recip_z);

        let tnear = max_m256(
            max_m256(min_m256(tmin_x, tmax_x), min_m256(tmin_y, tmax_y)),
            min_m256(tmin_z, tmax_z),
        );

        let tfar = min_m256(
            min_m256(max_m256(tmin_x, tmax_x), max_m256(tmin_y, tmax_y)),
            max_m256(tmin_z, tmax_z),
        );

        let hit = cmp_op_mask_m256::<{ cmp_op!(LessEqualOrdered) }>(
            max_m256(tnear, ray.near),
            min_m256(tfar, ray.far),
        );

        (
            WideBoolF32x8::from_arr(hit.to_array()),
            tnear.to_array().into(),
        )
    }
}
