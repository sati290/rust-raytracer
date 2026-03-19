use ultraviolet::{Vec3x4, Vec3x8};
use wide::{CmpGe as _, CmpLe as _, f32x4, f32x8};

use crate::triangle_opt::TriangleOpt;

macro_rules! triangle_intersector_n {
    ($(($name:ident, $scalar:ident, $vector:ident)),+) => {
        $(
            pub struct $name;

            impl $name {
                pub fn intersect(triangle: &TriangleOpt, ray_origin: &$vector, ray_direction: &$vector) -> $scalar {
                    let v0v1 = $vector::splat(triangle.v0v1);
                    let v0v2 = $vector::splat(triangle.v0v2);
                    let pvec = ray_direction.cross(v0v1);
                    let det = v0v2.dot(pvec);

                    let epsilon = $scalar::splat(f32::EPSILON);
                    let det_valid = det.abs().cmp_ge(epsilon);

                    let inv_det = 1. / det;

                    let tvec = *ray_origin - $vector::splat(triangle.v0);
                    let u = tvec.dot(pvec) * inv_det;
                    let u_valid = u.cmp_ge(0.) & u.cmp_le(1.);

                    let qvec = tvec.cross(v0v2);
                    let v = ray_direction.dot(qvec) * inv_det;
                    let v_valid = v.cmp_ge(0.) & (u + v).cmp_le(1.);

                    let t = v0v1.dot(qvec) * inv_det;

                    let t_valid = det_valid & u_valid & v_valid;
                    t_valid.blend(t, $scalar::splat(f32::INFINITY))
                }
            }
        )+
    }
}

triangle_intersector_n!(
    (TriangleIntersector4, f32x4, Vec3x4),
    (TriangleIntersector8, f32x8, Vec3x8)
);
