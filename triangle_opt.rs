use ultraviolet::{Vec3, Vec3x4};
use wide::{CmpGe, CmpLe, f32x4};

use crate::triangle::Triangle;

// Triangle optimized for intersection tests
pub struct TriangleOpt {
    pub v0v1: Vec3,
    pub v0v2: Vec3,
    pub v0: Vec3,
}

impl TriangleOpt {
    #[must_use]
    pub fn _intersect(&self, ray_origin: &Vec3, ray_direction: &Vec3) -> f32 {
        let pvec = ray_direction.cross(self.v0v1);
        let det = self.v0v2.dot(pvec);

        let epsilon = 0.0000001;
        if det < epsilon {
            return f32::INFINITY;
        }

        let inv_det = 1. / det;

        let tvec = *ray_origin - self.v0;
        let u = tvec.dot(pvec) * inv_det;
        if !(0. ..=1.).contains(&u) {
            return f32::INFINITY;
        }

        let qvec = tvec.cross(self.v0v2);
        let v = ray_direction.dot(qvec) * inv_det;
        if !(0. ..=1.).contains(&v) {
            return f32::INFINITY;
        }

        self.v0v1.dot(qvec) * inv_det
    }

    #[must_use]
    pub fn intersect_simd(&self, ray_origin: &Vec3x4, ray_direction: &Vec3x4) -> f32x4 {
        let v0v1 = Vec3x4::splat(self.v0v1);
        let v0v2 = Vec3x4::splat(self.v0v2);
        let pvec = ray_direction.cross(v0v1);
        let det = v0v2.dot(pvec);

        let epsilon = f32x4::splat(0.0000001);
        let det_valid = det.cmp_ge(epsilon);

        let inv_det = 1. / det;

        let tvec = *ray_origin - Vec3x4::splat(self.v0);
        let u = tvec.dot(pvec) * inv_det;
        let u_valid = u.cmp_ge(0.) & u.cmp_le(1.);

        let qvec = tvec.cross(v0v2);
        let v = ray_direction.dot(qvec) * inv_det;
        let v_valid = v.cmp_ge(0.) & v.cmp_le(1.);

        let t = v0v1.dot(qvec) * inv_det;

        let t_valid = det_valid & u_valid & v_valid;
        t_valid.blend(t, f32x4::splat(f32::INFINITY))
    }
}

impl From<&Triangle> for TriangleOpt {
    fn from(triangle: &Triangle) -> TriangleOpt {
        TriangleOpt {
            v0v1: triangle.verts[1] - triangle.verts[0],
            v0v2: triangle.verts[2] - triangle.verts[0],
            v0: triangle.verts[0],
        }
    }
}
