use std::marker::PhantomData;

use nalgebra::{Scalar, SimdBool, SimdRealField, SimdValue, Vector3};

use crate::triangle_opt::TriangleOpt;

pub struct TriangleIntersector<T> {
    _phantom: PhantomData<T>,
}

impl<T> TriangleIntersector<T>
where
    T: Copy + Scalar + SimdRealField<Element = f32>,
{
    #[must_use]
    #[inline]
    pub fn intersect(
        triangle: &TriangleOpt,
        ray_origin: &Vector3<T>,
        ray_direction: &Vector3<T>,
    ) -> T {
        let v0v1 = Vector3::<T>::splat(triangle.v0v1);
        let v0v2 = Vector3::<T>::splat(triangle.v0v2);
        let pvec = ray_direction.cross(&v0v1);
        let det = v0v2.dot(&pvec);

        let epsilon = T::splat(f32::EPSILON);
        let det_valid = det.simd_abs().simd_ge(epsilon);

        let inv_det = T::one() / det;

        let tvec = *ray_origin - Vector3::<T>::splat(triangle.v0);
        let u = tvec.dot(&pvec) * inv_det;
        let u_valid = u.simd_ge(T::zero()) & u.simd_le(T::one());

        let qvec = tvec.cross(&v0v2);
        let v = ray_direction.dot(&qvec) * inv_det;
        let v_valid = v.simd_ge(T::zero()) & (u + v).simd_le(T::one());

        let t = v0v1.dot(&qvec) * inv_det;

        let t_valid = det_valid & u_valid & v_valid;
        t_valid.if_else(|| t, || T::splat(f32::INFINITY))
    }
}
