use crate::Vec3;

pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
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
    pub fn grow(&self, point: Vec3) -> Aabb {
        Aabb {
            min: self.min.min_by_component(point),
            max: self.max.max_by_component(point),
        }
    }

    pub fn grow_mut(&mut self, point: Vec3) {
        self.min = self.min.min_by_component(point);
        self.max = self.max.max_by_component(point);
    }

    #[must_use]
    pub fn join(&self, other: Aabb) -> Aabb {
        Aabb {
            min: self.min.min_by_component(other.min),
            max: self.max.max_by_component(other.max),
        }
    }

    pub fn join_mut(&mut self, other: Aabb) {
        self.min = self.min.min_by_component(other.min);
        self.max = self.max.max_by_component(other.max);
    }
}
