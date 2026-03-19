use ultraviolet::{Bivec3x4, Bivec3x8, Rotor3x4, Rotor3x8};
use wide::{f32x4, f32x8};

pub trait BlendRotor<T> {
    #[must_use]
    fn blend_rotor(&self, a: &T, b: &T) -> T;
}

macro_rules! impl_blend_rotor3 {
    ($(($t:ident, $rt:ident, $bt:ident)),+) => {
        $(
            impl BlendRotor<$rt> for $t {
                #[inline]
                fn blend_rotor(&self, a: &$rt, b: &$rt) -> $rt {
                    $rt {
                        s: self.blend(a.s, b.s),
                        bv: $bt {
                            xy: self.blend(a.bv.xy, b.bv.xy),
                            xz: self.blend(a.bv.xz, b.bv.xz),
                            yz: self.blend(a.bv.yz, b.bv.yz),
                        },
                    }
                }
            }
        )+
    }
}

impl_blend_rotor3!((f32x4, Rotor3x4, Bivec3x4), (f32x8, Rotor3x8, Bivec3x8));
