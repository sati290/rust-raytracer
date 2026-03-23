use crate::math::Vec3f;

#[derive(Clone)]
pub struct PointLight {
    pub pos: Vec3f,
    pub intensity: Vec3f,
}

impl PointLight {
    pub fn new(pos: Vec3f, color: Vec3f, power: f32) -> Self {
        PointLight {
            pos,
            intensity: color * power,
        }
    }
}
