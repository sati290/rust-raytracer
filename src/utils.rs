use ultraviolet::{Bivec3x4, Rotor3x4};
use wide::f32x4;

pub fn rotor_blend(a: Rotor3x4, b: Rotor3x4, mask: f32x4) -> Rotor3x4 {
    Rotor3x4 {
        s: mask.blend(a.s, b.s),
        bv: Bivec3x4 {
            xy: mask.blend(a.bv.xy, b.bv.xy),
            xz: mask.blend(a.bv.xz, b.bv.xz),
            yz: mask.blend(a.bv.yz, b.bv.yz),
        },
    }
}
