//! What a program is and where it runs: its encoding and its pipeline stage.
#[allow(unused_imports)]
use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

use serde::Serialize;

/// How a shader program's bytes are encoded. Only the encoding lives here; reading one is the job of the crate that
/// knows it (`iridia-dxbc`, `iridia-dxil`, `iridia-d3d9`, `iridia-mojo`), and the program's meaning is lifted from it on
/// demand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ShaderFormat {
    /// A DXBC container holding Shader Model 4 or 5 byte code.
    Dxbc,
    /// A DXBC container holding Shader Model 6 (DXIL, LLVM bitcode).
    Dxil,
    /// A SPIR-V module.
    SpirV,
    /// GLSL source text.
    Glsl,
    /// Raw Direct3D 9 shader tokens, including the version token and END (no DXBC container).
    D3d9,
}
/// The pipeline stage a program runs in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Stage {
    /// A vertex program: one invocation per vertex.
    Vertex,
    /// A pixel (fragment) program: one invocation per covered pixel.
    Pixel,
    /// A compute program: no varyings, system inputs naming the invocation, a thread group size.
    Compute,
    /// A hull program: the control-point phase and the patch-constant function of a tessellated patch (see
    /// [`super::StageFacts::Hull`]).
    Hull,
    /// A domain program: a vertex of a tessellated patch from its domain location (see [`super::StageFacts::Domain`]).
    Domain,
    /// A geometry program.
    Geometry,
    /// A mesh program: a thread group that writes the vertices and primitives of one meshlet (see
    /// [`super::StageFacts::Mesh`]); its outputs are the per-vertex varyings of the interface.
    Mesh,
    /// An amplification (task) program: a thread group that launches mesh thread groups, handing each the words of
    /// its task payload (see [`super::StageFacts::Amplification`]). It has no varyings.
    Amplification,
    /// A ray generation program: the start of a ray tracing dispatch, one invocation per dispatched ray position.
    RayGeneration,
    /// An intersection program: tests a ray against a procedural primitive and reports hits.
    Intersection,
    /// An any-hit program: decides whether a candidate hit counts.
    AnyHit,
    /// A closest-hit program: runs for the nearest accepted hit of a traced ray.
    ClosestHit,
    /// A miss program: runs when a traced ray hits nothing.
    Miss,
    /// A callable program: run by another ray tracing program (`CallShader`).
    Callable,
}
impl Stage {
    /// Whether a program of this stage runs in thread groups (`numthreads`): a compute, mesh or amplification
    /// program, whose invocations share workgroup memory and meet at a workgroup barrier.
    pub fn has_thread_groups(self) -> bool {
        matches!(self, Self::Compute | Self::Mesh | Self::Amplification)
    }

    /// Whether this is one of the ray tracing pipeline's stages: a program of it is functions that read the ray's
    /// system values and carry a payload, not an entry point (see `iridia::stage_functions`).
    pub fn is_ray_tracing(self) -> bool {
        matches!(
            self,
            Self::RayGeneration
                | Self::Intersection
                | Self::AnyHit
                | Self::ClosestHit
                | Self::Miss
                | Self::Callable
        )
    }

    /// The stage's name after the article it takes, for messages: `an Amplification`, `a Pixel`
    /// (`format!("{} program", stage.with_article())`).
    pub fn with_article(self) -> impl core::fmt::Display {
        WithArticle(self)
    }
}

/// [`Stage::with_article`]'s value.
struct WithArticle(Stage);

impl core::fmt::Display for WithArticle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let article = match self.0 {
            Stage::Amplification | Stage::Intersection | Stage::AnyHit => "an",
            _ => "a",
        };
        write!(f, "{article} {:?}", self.0)
    }
}
