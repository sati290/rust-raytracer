use crate::aabb::Aabb;
use ultraviolet::Vec3;

pub struct Triangle {
    pub verts: [Vec3; 3],
    pub normal: Vec3,
}

impl Triangle {
    #[must_use]
    pub fn new(verts: [Vec3; 3], normal: Vec3) -> Self {
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
}
