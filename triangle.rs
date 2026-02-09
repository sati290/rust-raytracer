use crate::aabb::Aabb;
use ultraviolet::{Vec3, Vec3x4};
use wide::{f32x4, CmpLe, CmpGe};

pub struct Triangle {
    pub verts: [Vec3; 3],
    pub normal: Vec3,
}

impl Triangle {
    #[must_use]
    pub fn new(verts: [Vec3; 3]) -> Self {
        let v0v1 = verts[1] - verts[0];
        let v0v2 = verts[2] - verts[0];
        let normal = v0v2.cross(v0v1).normalized();

        Triangle { verts, normal }
    }

    #[must_use]
    pub fn aabb(&self) -> Aabb {
        let mut aabb = Aabb::empty();

        for v in self.verts {
            aabb.grow_mut(v);
        }

        aabb
    }

    #[must_use]
    pub fn centroid(&self) -> Vec3 {
        (self.verts[0] + self.verts[1] + self.verts[2]) / 3.
    }

    #[must_use]
    pub fn intersect<const B: bool>(&self, ray_origin: &Vec3, ray_direction: &Vec3) -> Option<f32> {
        let v0v1 = self.verts[1] - self.verts[0];
        let v0v2 = self.verts[2] - self.verts[0];
        let pvec = ray_direction.cross(v0v1);
        let det = v0v2.dot(pvec);

        let epsilon = 0.0000001;
        if det < epsilon {
            return None;
        }

        let inv_det = 1. / det;

        let tvec = *ray_origin - self.verts[0];
        let u = tvec.dot(pvec) * inv_det;
        if !(0. ..=1.).contains(&u) {
            return None;
        }

        let qvec = tvec.cross(v0v2);
        let v = ray_direction.dot(qvec) * inv_det;
        if !(0. ..=1.).contains(&v) {
            return None;
        }

        let t = v0v1.dot(qvec) * inv_det;

        Some(t)
    }

    #[must_use]
    pub fn intersect_simd<const B: bool>(&self, ray_origin: &Vec3x4, ray_direction: &Vec3x4) -> f32x4 {
        let v0v1 = Vec3x4::splat(self.verts[1] - self.verts[0]);
        let v0v2 = Vec3x4::splat(self.verts[2] - self.verts[0]);
        let pvec = ray_direction.cross(v0v1);
        let det = v0v2.dot(pvec);

        let epsilon = f32x4::splat(0.0000001);
        let det_valid = det.cmp_ge(epsilon);

        let inv_det = 1. / det;

        let tvec = *ray_origin - Vec3x4::splat(self.verts[0]);
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
