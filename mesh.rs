use std::{collections::HashMap, hash::Hash, iter::Enumerate};

use crate::aabb::Aabb;
use ultraviolet::{Vec2, Vec3};

#[derive(Default)]
pub struct TriangleMesh {
    positions: Vec<Vec3>,
    texcoords: Vec<Vec2>,
    normals: Vec<Vec3>,
    indices: Vec<[u32; 3]>,
    geom_normals: Vec<Vec3>,
}

impl TriangleMesh {
    pub fn num_triangles(&self) -> usize {
        self.indices.len()
    }

    pub fn _num_vertices(&self) -> usize {
        self.positions.len()
    }

    pub fn get_triangle(&self, index: u32) -> Triangle<'_> {
        Triangle {
            mesh: self,
            index,
            vtx_indices: self.indices[index as usize],
        }
    }

    pub fn iter(&self) -> Iter<'_> {
        Iter::new(self)
    }
}

pub struct Iter<'a> {
    mesh: &'a TriangleMesh,
    iter: Enumerate<std::slice::Iter<'a, [u32; 3]>>,
}

impl<'a> Iter<'a> {
    fn new(mesh: &'a TriangleMesh) -> Self {
        Iter {
            mesh,
            iter: mesh.indices.iter().enumerate(),
        }
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = Triangle<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(index, vtx_indices)| Triangle {
            mesh: self.mesh,
            index: index as u32,
            vtx_indices: *vtx_indices,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

pub struct Triangle<'a> {
    mesh: &'a TriangleMesh,
    index: u32,
    vtx_indices: [u32; 3],
}

impl Triangle<'_> {
    pub fn vertices(&self) -> [&Vec3; 3] {
        self.vtx_indices
            .map(|i| unsafe { self.mesh.positions.get_unchecked(i as usize) })
    }

    pub fn normal(&self) -> &Vec3 {
        unsafe { self.mesh.geom_normals.get_unchecked(self.index as usize) }
    }

    #[must_use]
    pub fn aabb(&self) -> Aabb {
        let mut aabb = Aabb::empty();
        for v in self.vertices() {
            aabb.grow_mut(v);
        }

        aabb
    }

    #[must_use]
    pub fn centroid(&self) -> Vec3 {
        let verts = self.vertices();
        (*verts[0] + *verts[1] + *verts[2]) / 3.
    }
}

#[derive(PartialEq, Eq, Hash)]
struct VertexKey {
    data: [u32; 8],
}

impl VertexKey {
    pub fn new(p: &Vec3, t: &Vec2, n: &Vec3) -> Self {
        VertexKey {
            data: [
                p.x.to_bits(),
                p.y.to_bits(),
                p.z.to_bits(),
                t.x.to_bits(),
                t.y.to_bits(),
                n.x.to_bits(),
                n.y.to_bits(),
                n.z.to_bits(),
            ],
        }
    }
}

#[derive(Default)]
pub struct TriangleMeshBuilder {
    mesh: TriangleMesh,
    vertex_idx_map: HashMap<VertexKey, u32>,
}

impl TriangleMeshBuilder {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn reserve_vertices(&mut self, additional: usize) {
        self.mesh.positions.reserve(additional);
        self.mesh.texcoords.reserve(additional);
        self.mesh.normals.reserve(additional);
        self.vertex_idx_map.reserve(additional);
    }

    pub fn reserve_triangles(&mut self, additional: usize) {
        self.mesh.indices.reserve(additional);
        self.mesh.geom_normals.reserve(additional);
    }

    pub fn add_triangle(
        &mut self,
        positions: &[Vec3; 3],
        texcoords: &[Vec2; 3],
        normals: &[Vec3; 3],
    ) {
        let indices = [0, 1, 2].map(|i| {
            let p = &positions[i];
            let t = &texcoords[i];
            let n = &normals[i];

            let key = VertexKey::new(p, t, n);
            self.vertex_idx_map.get(&key).copied().unwrap_or_else(|| {
                let index = self.mesh.positions.len() as u32;
                self.mesh.positions.push(*p);
                self.mesh.texcoords.push(*t);
                self.mesh.normals.push(*n);
                self.vertex_idx_map.insert(key, index);
                index
            })
        });

        let mut ng = (positions[1] - positions[0])
            .cross(positions[2] - positions[0])
            .normalized();
        if ng.dot(normals[0]) < 0. {
            ng = -ng;
        }

        self.mesh.indices.push(indices);
        self.mesh.geom_normals.push(ng);
    }

    pub fn build(self) -> TriangleMesh {
        self.mesh
    }
}
