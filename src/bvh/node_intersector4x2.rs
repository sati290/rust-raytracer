use safe_arch::*;
use ultraviolet::Vec3x4;
use wide::f32x4;

use crate::bvh::simd_ray::SimdRay4x2Interleaved;

pub struct BvhNodeIntersector4x2 {}

impl BvhNodeIntersector4x2 {
    pub fn intersect(
        child_bbox: &Vec3x4,
        ray: &SimdRay4x2Interleaved,
    ) -> (f32x4, f32x4, f32x4, f32x4) {
        let hit_shuffle = set_i32_m256i(7, 5, 3, 1, 6, 4, 2, 0);

        // R1   R1   R2   R2   R3   R3   R4   R4
        // MINL MINR MAXL MAXR MINL MINR MAXL MAXR
        let bb_min_max_x = load_m128_splat_m256(&m128::from(child_bbox.x.to_array()));
        let bb_min_max_y = load_m128_splat_m256(&m128::from(child_bbox.y.to_array()));
        let bb_min_max_z = load_m128_splat_m256(&m128::from(child_bbox.z.to_array()));

        // MINL MINR MINL MINR MINL MINR MINL MINR
        let bb_min_x = shuffle_m256::<0b01_00_01_00>(bb_min_max_x, bb_min_max_x);
        let bb_min_y = shuffle_m256::<0b01_00_01_00>(bb_min_max_y, bb_min_max_y);
        let bb_min_z = shuffle_m256::<0b01_00_01_00>(bb_min_max_z, bb_min_max_z);

        let bb_max_x = shuffle_m256::<0b11_10_11_10>(bb_min_max_x, bb_min_max_x);
        let bb_max_y = shuffle_m256::<0b11_10_11_10>(bb_min_max_y, bb_min_max_y);
        let bb_max_z = shuffle_m256::<0b11_10_11_10>(bb_min_max_z, bb_min_max_z);

        let tmin_x = fused_mul_sub_m256(bb_min_x, ray.dir_recip_x, ray.origin_dir_recip_x);
        let tmin_y = fused_mul_sub_m256(bb_min_y, ray.dir_recip_y, ray.origin_dir_recip_y);
        let tmin_z = fused_mul_sub_m256(bb_min_z, ray.dir_recip_z, ray.origin_dir_recip_z);
        let tmax_x = fused_mul_sub_m256(bb_max_x, ray.dir_recip_x, ray.origin_dir_recip_x);
        let tmax_y = fused_mul_sub_m256(bb_max_y, ray.dir_recip_y, ray.origin_dir_recip_y);
        let tmax_z = fused_mul_sub_m256(bb_max_z, ray.dir_recip_z, ray.origin_dir_recip_z);

        // R1    R1    R2    R2    R3    R3    R4    R4
        // NEARL NEARR NEARL NEARR NEARL NEARR NEARL NEARR
        let tnear = max_m256(
            max_m256(min_m256(tmin_x, tmax_x), min_m256(tmin_y, tmax_y)),
            min_m256(tmin_z, tmax_z),
        );

        let tfar = min_m256(
            min_m256(max_m256(tmin_x, tmax_x), max_m256(tmin_y, tmax_y)),
            max_m256(tmin_z, tmax_z),
        );

        let hit = shuffle_av_f32_all_m256(
            cmp_op_mask_m256::<{ cmp_op!(LessEqualOrdered) }>(
                max_m256(tnear, ray.near),
                min_m256(tfar, ray.far),
            ),
            hit_shuffle,
        );

        let hit_l = f32x4::from(cast_to_m128_from_m256(hit).to_array());
        let hit_r = f32x4::from(extract_m128_from_m256::<1>(hit).to_array());

        let near = shuffle_av_f32_all_m256(tnear, hit_shuffle);
        let near_l = f32x4::from(cast_to_m128_from_m256(near).to_array());
        let near_r = f32x4::from(extract_m128_from_m256::<1>(near).to_array());

        (hit_l, hit_r, near_l, near_r)
    }
}
