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

    pub fn intersect(&self, ray_origin: &Vec3, ray_direction_recip: &Vec3) -> bool {
        let tx1 = (self.min.x - ray_origin.x) * ray_direction_recip.x;
        let tx2 = (self.max.x - ray_origin.x) * ray_direction_recip.x;

        let tmin = tx1.min(tx2);
        let tmax = tx1.max(tx2);

        let ty1 = (self.min.y - ray_origin.y) * ray_direction_recip.y;
        let ty2 = (self.max.y - ray_origin.y) * ray_direction_recip.y;

        let tmin = tmin.max(ty1.min(ty2));
        let tmax = tmax.min(ty1.max(ty2));

        let tz1 = (self.min.z - ray_origin.z) * ray_direction_recip.z;
        let tz2 = (self.max.z - ray_origin.z) * ray_direction_recip.z;

        let tmin = tmin.max(tz1.min(tz2));
        let tmax = tmax.min(tz1.max(tz2));

        tmax >= tmin.max(0.)
    }
}
