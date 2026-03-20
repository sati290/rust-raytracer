use core::f32;

use rand::Rng;
use ultraviolet::{Vec3, Vec3x4, Vec3x8, Vec4};
use wide::{CmpLt, f32x4, f32x8};

use crate::{
    Tile,
    bvh::{BvhIntersector4, BvhIntersector8},
    camera::CameraRayGenerator,
    integrators::common::{IntegratorsCommon4, IntegratorsCommon8, SHADOW_RAY_NEAR},
    ray::{Ray4, Ray8, RayHit4, RayHit8},
    scene::Scene,
    trace_stats::TraceStats,
};

macro_rules! tile_integrator_n {
    ($(($n:ident, $c:literal, $t:ident, $vt:ident, $r:ident, $rh:ident, $common:ident, $intersector:ident, $sample:ident)),+) => {
        $(
            pub struct $n;

            impl $n {
                fn integrate_ray<R: Rng>(
                    camera_ray: $rh,
                    scene: &Scene,
                    max_bounces: u8,
                    dest_pixel: &mut Vec4,
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
                    let mut weight = $vt::one();
                    let mut ray_hit = camera_ray;
                    let mut ray_valid = $t::splat(-0.);

                    loop {
                        $intersector::intersect(bvh, &mut ray_hit, trace_stats);
                        ray_valid &= ray_hit.ray.far.cmp_lt(f32::INFINITY);
                        if ray_valid.none() {
                            break;
                        }

                        let mut normal = [Vec3::zero(); $c];
                        for (i, h) in ray_hit.obj_idx.into_iter().enumerate() {
                            if let Some(h) = h {
                                let hit_obj = mesh.get_triangle(h);
                                normal[i] = *hit_obj.normal();
                            }
                        }

                        let hit_pos = ray_hit.ray.hit_pos();
                        let dir_out = -ray_hit.ray.direction;
                        let normal = $vt::from(normal);

                        // Shadow ray
                        let (shadow_ray_dir, shadow_far, shadow_weight, shadow_valid) =
                            $common::sample_light(&dir_out, &normal, &hit_pos, &ray_valid, light);
                        if shadow_valid.any() {
                            let shadow_ray = $r::new(
                                &hit_pos,
                                &shadow_ray_dir,
                                &$t::splat(SHADOW_RAY_NEAR),
                                &shadow_far,
                                &shadow_valid,
                            );
                            let occluded = $intersector::occluded(bvh, &shadow_ray, trace_stats);
                            let light_valid = shadow_valid & !occluded;
                            let contrib: [Vec3; $c] = $vt::blend(light_valid, weight * shadow_weight, $vt::zero()).into();
                            for c in contrib.into_iter() {
                                *dest_pixel += Vec4::from(c);
                            }
                        }

                        // Diffuse ray
                        if bounces >= max_bounces {
                            break;
                        }

                        let (dir_in, diffuse_weight) =
                            $common::sample_diffuse_ray(&dir_out, &normal, rng);
                        ray_hit = $r::new(
                            &hit_pos,
                            &dir_in,
                            &$t::splat(SHADOW_RAY_NEAR),
                            &$t::splat(f32::INFINITY),
                            &ray_valid,
                        )
                        .into();
                        bounces += 1;
                        weight *= diffuse_weight;
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

                    let cam_pos4 = $vt::splat(scene.camera.position());
                    let mut ray_generator =
                        CameraRayGenerator::new(&scene.camera, viewport_size.0, viewport_size.1, tile.region);

                    while !ray_generator.is_done() {
                        let dest_pixel = &mut tile.pixels[ray_generator.current_pixel_idx() as usize];
                        for _ in 0..samples.div_ceil($c) {
                            let ray_dir = ray_generator.$sample(&mut tile.rng);

                            Self::integrate_ray(
                                $r::new(
                                    &cam_pos4,
                                    &ray_dir,
                                    &$t::ZERO,
                                    &$t::splat(f32::INFINITY),
                                    &$t::splat(-0.),
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
        )+
    }
}

tile_integrator_n!(
    (
        TileIntegrator4,
        4,
        f32x4,
        Vec3x4,
        Ray4,
        RayHit4,
        IntegratorsCommon4,
        BvhIntersector4,
        sample4
    ),
    (
        TileIntegrator8,
        8,
        f32x8,
        Vec3x8,
        Ray8,
        RayHit8,
        IntegratorsCommon8,
        BvhIntersector8,
        sample8
    )
);
