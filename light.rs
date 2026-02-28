use ultraviolet::Vec3;

pub struct PointLight {
    pub pos: Vec3,
    pub intensity: Vec3,
}

impl PointLight {
    pub fn new(pos: Vec3, color: Vec3, power: f32) -> Self {
        PointLight {
            pos,
            intensity: color * power,
        }
    }
}
