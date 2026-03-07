use std::{sync::LazyLock, time::Instant};

use obj::Obj;
use ultraviolet::{Mat3, Vec2, Vec3};

use crate::{
    bvh::Bvh,
    camera::Camera,
    light::PointLight,
    mesh::{TriangleMesh, TriangleMeshBuilder},
};

pub struct SceneDefinition {
    obj_path: String,
    transform: Mat3,
    camera: Camera,
    light: PointLight,
}

pub struct Scene {
    pub mesh: TriangleMesh,
    pub bvh: Bvh,
    pub camera: Camera,
    pub light: PointLight,
}

impl Scene {
    pub fn load(def: &SceneDefinition) -> Self {
        let load_start = Instant::now();
        let obj = Obj::load(&def.obj_path).unwrap();

        let mut skipped_zero_area: u32 = 0;
        let mut builder = TriangleMeshBuilder::new();
        builder.reserve_vertices(obj.data.position.len());

        for o in &obj.data.objects {
            for g in &o.groups {
                builder.reserve_triangles(g.polys.len());
                for p in &g.polys {
                    assert!(p.0.len() >= 3);

                    for i in 0..(p.0.len() - 2) {
                        let indices = [0, i + 1, i + 2];
                        let positions = indices
                            .map(|i| def.transform * Vec3::from(obj.data.position[p.0[i].0]));
                        let texcoords =
                            indices.map(|i| Vec2::from(obj.data.texture[p.0[i].1.unwrap()]));
                        let normals = indices.map(|i| {
                            def.transform * Vec3::from(obj.data.normal[p.0[i].2.unwrap()])
                        });

                        let v0v1 = positions[1] - positions[0];
                        let v0v2 = positions[2] - positions[0];
                        let magsq = v0v1.cross(v0v2).mag_sq();
                        if magsq == 0. {
                            skipped_zero_area += 1;
                            continue;
                        }

                        builder.add_triangle(&positions, &texcoords, &normals);
                    }
                }
            }
        }

        let mesh = builder.build();

        let elapsed = load_start.elapsed();
        println!(
            "loaded {}, {} triangles in {:.2?}",
            def.obj_path,
            mesh.num_triangles(),
            elapsed
        );
        if skipped_zero_area > 0 {
            println!("skipped {} triangles with zero area", skipped_zero_area);
        }

        let bvh_start = Instant::now();
        let bvh = Bvh::build(&mesh);
        println!("bvh build {:.2?}", bvh_start.elapsed());

        Scene {
            mesh,
            bvh,
            camera: def.camera.clone(),
            light: def.light.clone(),
        }
    }
}

pub static SCENE_ASIAN_DRAGON: LazyLock<SceneDefinition> = LazyLock::new(|| SceneDefinition {
    obj_path: String::from("./scenes/asian_dragon_obj/asian_dragon.obj"),
    transform: Mat3::new(
        Vec3::new(1. / 1000., 0., 0.),
        Vec3::new(0., 0., 1. / 1000.),
        Vec3::new(0., 1. / 1000., 0.),
    )
    .transposed(),
    camera: Camera::new(
        Vec3::new(0.6, -1., 0.25).normalized() * 2.5,
        Vec3::new(0., 0., 0.35),
        Vec3::unit_z(),
        60.,
    ),
    light: PointLight::new(Vec3::new(5., -10., 5.), Vec3::one(), 300.),
});

pub static SCENE_SANMIGUEL: LazyLock<SceneDefinition> = LazyLock::new(|| SceneDefinition {
    obj_path: String::from("./scenes/San_Miguel/san-miguel.obj"),
    transform: Mat3::new(
        Vec3::new(1., 0., 0.),
        Vec3::new(0., 0., -1.),
        Vec3::new(0., 1., 0.),
    )
    .transposed(),
    camera: Camera::new(
        Vec3::new(28., -1.65, 1.8),
        Vec3::new(27., -1.65, 1.8),
        Vec3::unit_z(),
        60.,
    ),
    light: PointLight::new(Vec3::new(20., -2.5, 3.), Vec3::one(), 20.),
});
