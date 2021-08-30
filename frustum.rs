use crate::Ray;
use ultraviolet::{Mat4, Vec3, Vec3x4, Vec4};

#[derive(Clone, Copy)]
pub struct Frustum {
    pub normals: [Vec3; 4],
    pub offsets: [f32; 4],
    pub normals_optimized: [Vec3x4; 2],
}

impl Frustum {
    fn new(normals: &[Vec3; 4], offsets: &[f32; 4]) -> Self {
        let normals_optimized = Vec3x4::from(*normals);
        let normals_optimized = [
            normals_optimized.max_by_component(Vec3x4::zero()),
            normals_optimized.min_by_component(Vec3x4::zero()),
        ];

        Frustum {
            normals: *normals,
            offsets: *offsets,
            normals_optimized,
        }
    }

    pub fn from_corner_rays(rays: &[Ray; 4]) -> Self {
        let mut normals: [Vec3; 4] = Default::default();
        let mut offsets: [f32; 4] = Default::default();

        for i in 0..4 {
            normals[i] = rays[i]
                .direction
                .cross(rays[(i + 1) % 4].direction)
                .normalized();
            offsets[i] = rays[i].origin.dot(normals[i]);
        }

        Frustum::new(&normals, &offsets)
    }
}

impl std::ops::Mul<Frustum> for Mat4 {
    type Output = Frustum;

    fn mul(self, rhs: Frustum) -> Self::Output {
        let mut normals: [Vec3; 4] = Default::default();
        let mut offsets: [f32; 4] = Default::default();

        let mat = self.adjugate().transposed();

        for i in 0..4 {
            let v = mat
                * Vec4::new(
                    rhs.normals[i].x,
                    rhs.normals[i].y,
                    rhs.normals[i].z,
                    -rhs.offsets[i],
                );
            normals[i] = v.xyz();
            offsets[i] = -v.w;
        }

        Frustum::new(&normals, &offsets)
    }
}
