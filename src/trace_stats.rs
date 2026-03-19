use std::ops::Add;

#[derive(Clone, Copy, Debug)]
pub struct TraceStats {
    total_rays: u64,
    inner_visit: u64,
    inner_rays: u64,
    leaf_visit: u64,
    leaf_rays: u64,
    leaf_objs: u64,
    obj_intersect: u64,
}

impl TraceStats {
    pub fn new() -> Self {
        TraceStats {
            total_rays: 0,
            inner_visit: 0,
            inner_rays: 0,
            leaf_visit: 0,
            leaf_rays: 0,
            leaf_objs: 0,
            obj_intersect: 0,
        }
    }

    #[inline]
    pub fn trace_start(&mut self, total_rays: u64) {
        self.total_rays += total_rays;
    }

    #[inline]
    pub fn inner_visit(&mut self, rays: u64) {
        self.inner_visit += 1;
        self.inner_rays += rays;
    }

    #[inline]
    pub fn leaf_visit(&mut self, rays: u64, objects: u64) {
        self.leaf_visit += 1;
        self.leaf_rays += rays;
        self.leaf_objs += objects;
        self.obj_intersect += rays * objects;
    }

    #[inline]
    pub fn total_rays(&self) -> u64 {
        self.total_rays
    }

    pub fn print(&self) {
        println!("{:?}", self);
        println!(
            "avg {:.1} rays per inner visit",
            self.inner_rays as f32 / self.inner_visit as f32
        );
        println!(
            "avg {:.1} rays {:.1} objs per leaf visit",
            self.leaf_rays as f32 / self.leaf_visit as f32,
            self.leaf_objs as f32 / self.leaf_visit as f32
        );
        println!(
            "avg {:.1} intersects per ray",
            self.obj_intersect as f32 / self.total_rays as f32
        );
    }
}

impl Add for TraceStats {
    type Output = TraceStats;

    fn add(self, rhs: Self) -> Self::Output {
        TraceStats {
            total_rays: self.total_rays + rhs.total_rays,
            inner_visit: self.inner_visit + rhs.inner_visit,
            inner_rays: self.inner_rays + rhs.inner_rays,
            leaf_visit: self.leaf_visit + rhs.leaf_visit,
            leaf_rays: self.leaf_rays + rhs.leaf_rays,
            leaf_objs: self.leaf_objs + rhs.leaf_objs,
            obj_intersect: self.obj_intersect + rhs.obj_intersect,
        }
    }
}
