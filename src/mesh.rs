use std::{collections::HashMap, hash::Hash, iter::Enumerate};

use crate::{
    aabb::Aabb,
    math::{Vec2f, Vec3f},
};

#[derive(Default)]
pub struct TriangleMesh {
    positions: Vec<Vec3f>,
    texcoords: Vec<Vec2f>,
    normals: Vec<Vec3f>,
    indices: Vec<[u32; 3]>,
    geom_normals: Vec<Vec3f>,
}

impl TriangleMesh {
    #[must_use]
    #[inline]
    pub fn num_triangles(&self) -> usize {
        self.indices.len()
    }

    #[must_use]
    #[inline]
    pub fn _num_vertices(&self) -> usize {
        self.positions.len()
    }

    #[must_use]
    #[inline]
    pub fn get_triangle(&self, index: u32) -> Triangle<'_> {
        Triangle {
            mesh: self,
            index,
            vtx_indices: self.indices[index as usize],
        }
    }

    #[must_use]
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
    #[must_use]
    #[inline]
    pub fn vertices(&self) -> [&Vec3f; 3] {
        self.vtx_indices
            .map(|i| unsafe { self.mesh.positions.get_unchecked(i as usize) })
    }

    #[must_use]
    #[inline]
    pub fn normal(&self) -> &Vec3f {
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
    pub fn centroid(&self) -> Vec3f {
        let verts = self.vertices();
        (*verts[0] + *verts[1] + *verts[2]) / 3.
    }
}

#[derive(PartialEq, Eq, Hash)]
struct VertexKey {
    data: [u32; 8],
}

impl VertexKey {
    pub fn new(p: &Vec3f, t: &Vec2f, n: &Vec3f) -> Self {
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
        positions: &[Vec3f; 3],
        texcoords: &[Vec2f; 3],
        normals: &[Vec3f; 3],
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

        let v0v1 = positions[1] - positions[0];
        let v0v2 = positions[2] - positions[0];
        let mut ng = (v0v1).cross(&v0v2).normalize();
        if ng.dot(&normals[0]) < 0. {
            ng = -ng;
        }

        self.mesh.indices.push(indices);
        self.mesh.geom_normals.push(ng);
    }

    #[must_use]
    pub fn build(self) -> TriangleMesh {
        self.mesh
    }
}
