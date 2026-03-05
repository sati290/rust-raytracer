use std::time::Instant;

use obj::Obj;
use ultraviolet::Vec3;

use crate::{bvh::Bvh, camera::Camera, light::PointLight, triangle::Triangle};

pub struct Scene {
    pub objects: Vec<Triangle>,
    pub bvh: Bvh,
    pub camera: Camera,
    pub light: PointLight,
}

impl Scene {
    pub fn load(obj_path: &str, camera: Camera, light: PointLight) -> Self {
        let load_start = Instant::now();
        let obj = Obj::load(obj_path).unwrap();

        let mut skipped_zero_area: u32 = 0;
        let mut objects = Vec::new();
        for o in &obj.data.objects {
            for g in &o.groups {
                for p in &g.polys {
                    assert!(p.0.len() >= 3);

                    for i in 0..(p.0.len() - 2) {
                        let verts = [
                            Vec3::from(obj.data.position[p.0[0].0]) * Vec3::new(1., 1., -1.),
                            Vec3::from(obj.data.position[p.0[i + 1].0]) * Vec3::new(1., 1., -1.),
                            Vec3::from(obj.data.position[p.0[i + 2].0]) * Vec3::new(1., 1., -1.),
                        ];

                        let v0v1 = verts[1] - verts[0];
                        let v0v2 = verts[2] - verts[0];
                        let normal = v0v2.cross(v0v1).normalized();

                        let magsq = v0v1.cross(v0v2).mag_sq();
                        if magsq == 0. {
                            skipped_zero_area += 1;
                            continue;
                        }

                        let tri = Triangle::new(verts, normal);
                        objects.push(tri);
                    }
                }
            }
        }

        let elapsed = load_start.elapsed();
        println!(
            "loaded {}, {} triangles in {:.2?}",
            obj_path,
            objects.len(),
            elapsed
        );
        if skipped_zero_area > 0 {
            println!("skipped {} triangles with zero area", skipped_zero_area);
        }

        let bvh_start = Instant::now();
        let bvh = Bvh::build(&objects);
        println!("bvh build {:.2?}", bvh_start.elapsed());

        Scene {
            objects,
            bvh,
            camera,
            light,
        }
    }

    pub fn load_asian_dragon(image_width: u32, image_height: u32) -> Self {
        let cam_pos = Vec3::new(0.6, 0.25, -1.).normalized() * 2500.;
        let cam_target = Vec3::new(0., 350., 0.);
        let light = PointLight::new(Vec3::new(5000., 5000., -10000.), Vec3::one(), 3e8);

        let camera = Camera::new(
            cam_pos,
            cam_target,
            Vec3::unit_y(),
            60.,
            image_width,
            image_height,
        );
        Self::load("./scenes/asian_dragon_obj/asian_dragon.obj", camera, light)
    }

    pub fn load_san_miguel(image_width: u32, image_height: u32) -> Self {
        let cam_pos = Vec3::new(28., 1.8, -1.65);
        //let cam_pos = Vec3::new(17., 2., -1.8);
        let cam_target = cam_pos + Vec3::new(-1., 0., 0.);
        //let light = PointLight::new(Vec3::new(20., 3., -1.65), Vec3::one(), 10.);
        let light = PointLight::new(Vec3::new(20., 3., -2.5), Vec3::one(), 20.);

        let camera = Camera::new(
            cam_pos,
            cam_target,
            Vec3::unit_y(),
            60.,
            image_width,
            image_height,
        );
        Self::load("./scenes/San_Miguel/san-miguel.obj", camera, light)
    }
}
