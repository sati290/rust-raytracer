use core::f32;
use std::marker::PhantomData;

use nalgebra::{SimdBool, SimdRealField, SimdValue, Vector3, vector};
use rand::{Rng, distributions::Standard, prelude::Distribution};

use crate::{
    Tile,
    bvh::{BvhIntersector, BvhNodeIntersectorType},
    camera::CameraRayGenerator,
    integrators::common::{IntegratorsCommon, SHADOW_RAY_NEAR},
    math::{SimdBoolSplat, SimdType, Vec4f},
    ray::{Ray, RayHit},
    scene::Scene,
    trace_stats::TraceStats,
};

pub struct TileIntegrator<T> {
    _phantom: PhantomData<T>,
}

impl<T> TileIntegrator<T>
where
    T: SimdRealField<Element = f32> + SimdType + BvhNodeIntersectorType + Copy,
    Standard: Distribution<T>,
    <T as SimdValue>::SimdBool: SimdBoolSplat,
{
    fn integrate_ray<R: Rng>(
        camera_ray: RayHit<T>,
        scene: &Scene,
        max_bounces: u8,
        dest_pixel: &mut Vec4f,
        rng: &mut R,
        trace_stats: &mut TraceStats,
    ) {
        let Scene {
            mesh,
            bvh,
            camera: _,
            light,
        } = scene;

        let mut bounces = 0;
        let mut weight = Vector3::<T>::from_element(T::one());
        let mut ray_hit = camera_ray;
        let mut ray_valid = T::SimdBool::splat(true);

        loop {
            BvhIntersector::<T>::intersect(bvh, &mut ray_hit, trace_stats);
            ray_valid = ray_valid & ray_hit.ray.far.simd_lt(T::splat(f32::INFINITY));
            if ray_valid.none() {
                break;
            }

            let mut normal = Vector3::<T>::zeros();
            for (i, h) in ray_hit.obj_idx.into_iter().enumerate() {
                if let Some(h) = h {
                    let hit_obj = mesh.get_triangle(h);
                    normal.replace(i, *hit_obj.normal());
                }
            }

            let hit_pos = ray_hit.ray.hit_pos();
            let dir_out = -ray_hit.ray.direction;

            // Shadow ray
            let (shadow_ray_dir, shadow_far, shadow_weight, shadow_valid) =
                IntegratorsCommon::<T>::sample_light(
                    &dir_out, &normal, &hit_pos, &ray_valid, light,
                );
            if shadow_valid.any() {
                let shadow_ray = Ray::<T>::new(
                    &hit_pos,
                    &shadow_ray_dir,
                    &T::splat(SHADOW_RAY_NEAR),
                    &shadow_far,
                    &shadow_valid,
                );
                let occluded = BvhIntersector::<T>::occluded(bvh, &shadow_ray, trace_stats);
                let light_valid = shadow_valid & !occluded;
                let contrib = light_valid
                    .if_else(|| weight.component_mul(&shadow_weight), Vector3::<T>::zeros);
                let c = contrib.map(|x| x.simd_horizontal_sum());
                *dest_pixel += vector![c.x, c.y, c.z, 0.];
            }

            // Diffuse ray
            if bounces >= max_bounces {
                break;
            }

            let (dir_in, diffuse_weight) =
                IntegratorsCommon::<T>::sample_diffuse_ray(&dir_out, &normal, rng);
            ray_hit = Ray::<T>::new(
                &hit_pos,
                &dir_in,
                &T::splat(SHADOW_RAY_NEAR),
                &T::splat(f32::INFINITY),
                &ray_valid,
            )
            .into();
            bounces += 1;
            weight.component_mul_assign(&diffuse_weight);
        }
    }

    pub fn integrate<R: Rng>(
        tile: &mut Tile<R>,
        viewport_size: (u32, u32),
        scene: &Scene,
        samples: u32,
        max_bounces: u8,
    ) {
        for p in &mut tile.pixels {
            p.w += samples as f32;
        }

        let cam_pos = Vector3::<T>::splat(scene.camera.position());
        let mut ray_generator =
            CameraRayGenerator::new(&scene.camera, viewport_size.0, viewport_size.1, tile.region);

        while !ray_generator.is_done() {
            let dest_pixel = &mut tile.pixels[ray_generator.current_pixel_idx() as usize];
            for _ in 0..samples.div_ceil(T::LANES as u32) {
                let ray_dir = ray_generator.sample::<T>(&mut tile.rng);

                Self::integrate_ray(
                    Ray::<T>::new(
                        &cam_pos,
                        &ray_dir,
                        &T::zero(),
                        &T::splat(f32::INFINITY),
                        &T::SimdBool::splat(true),
                    )
                    .into(),
                    scene,
                    max_bounces,
                    dest_pixel,
                    &mut tile.rng,
                    &mut tile.trace_stats,
                );
            }

            ray_generator.next_pixel();
        }
    }
}
