use clap::{Parser, ValueEnum};

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum TraceMode {
    Stream,
    StreamShadowImmediate,
    StreamCameraOnly,
    SingleRay,
    Packet4,
    Packet4x2,
    Packet8,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Scene {
    AsianDragon,
    SanMiguel,
}

#[derive(Parser, Debug)]
pub struct Args {
    #[arg(value_enum, short = 's', long, default_value_t = Scene::SanMiguel)]
    pub scene: Scene,

    #[arg(short = 'n', long, default_value_t = 128)]
    pub samples: u32,

    #[arg(short = 'b', long, default_value_t = 2)]
    pub max_bounces: u8,

    #[arg(short = 't', long, default_value_t = 16)]
    pub tile_size: u32,

    #[arg(value_enum, short = 'm', long, default_value_t = TraceMode::StreamShadowImmediate)]
    pub mode: TraceMode,

    #[arg(long, default_value_t = 1235468)]
    pub seed: u64,

    #[arg(long, default_value_t = false)]
    pub singlethread: bool,

    #[arg(long, short = 'p', default_value_t = false)]
    pub progress: bool,
}
