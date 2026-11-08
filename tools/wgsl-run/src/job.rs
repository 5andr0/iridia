//! The job format: one JSON object a line on standard input, one reply a line on standard output (README.md states
//! it with examples).

use serde::{Deserialize, Serialize};

/// One program to run.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Job {
    /// Echoed in the reply.
    #[serde(default)]
    pub id: serde_json::Value,
    /// The program's WGSL.
    pub wgsl: String,
    /// The entry point to run; the module's only one when absent.
    #[serde(default)]
    pub entry: Option<String>,
    /// A vertex or pixel program's invocations: each one's entry arguments in order, each argument's words (a scalar one,
    /// a vector one a lane, a bool 0 or 1, a 64-bit lane two, the low first; a structure its members' in order). A pixel
    /// program's run four at a time as the 2x2 quads of a rasterizer, invocation `4q + k` lane `k` of quad `q` (0 top
    /// left, 1 top right, 2 bottom left, 3 bottom right), so derivatives are taken between them.
    #[serde(default)]
    pub invocations: Vec<Vec<Vec<u32>>>,
    /// The thread groups a compute program is dispatched over.
    #[serde(default)]
    pub dispatch: Option<[u32; 3]>,
    /// The resources the entry point uses, by group and binding.
    #[serde(default)]
    pub resources: Vec<Resource>,
}

/// A resource of the program.
#[derive(Debug, Deserialize)]
pub struct Resource {
    /// Its bind group.
    pub group: u32,
    /// Its binding.
    pub binding: u32,
    /// What it holds.
    #[serde(flatten)]
    pub kind: ResourceKind,
}

/// What a resource holds.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceKind {
    /// A uniform or storage buffer: its words. A uniform buffer is as long as its type, the words past those given zero;
    /// a storage buffer is as long as the words given.
    Buffer {
        /// Its words.
        words: Vec<u32>,
    },
    /// A texture (sampled, depth or storage), every texel of every layer the same: `texel` its four lanes' bits (a depth
    /// texture's first lane its depth, an `r32` storage texture's first lane its value). One mip level.
    Texture {
        /// Each texel's lanes.
        texel: [u32; 4],
        /// Width, height and depth (a 3D texture's); 4 texels a side and a depth of 4 where absent.
        #[serde(default)]
        size: Option<[u32; 3]>,
        /// The layers of an arrayed texture (2 where absent; a cube's 6 a cube); a cube array's must be a multiple of 6
        /// (12 where absent).
        #[serde(default)]
        layers: Option<u32>,
        /// The samples per texel of a multisampled texture (a `texture_multisampled_2d` or a
        /// `texture_depth_multisampled_2d`; 4 where absent): a count the adapter supports for the texture's format.
        #[serde(default)]
        samples: Option<u32>,
        /// A multisampled texture's texel for each of its samples, in order (as many as `samples`); where absent every
        /// sample of every texel is `texel`.
        #[serde(default)]
        per_sample: Option<Vec<[u32; 4]>>,
    },
    /// A sampler: point filtering of the nearest level, coordinates clamped to the edge; a comparison sampler compares
    /// with `compare` (`less`, `less_equal`, `greater`, `greater_equal`, `equal`, `not_equal`, `always`, `never`).
    Sampler {
        /// A comparison sampler's function.
        #[serde(default)]
        compare: Option<String>,
    },
    /// A binding array's elements, in order, each a buffer, a texture or a sampler as the array's element type takes:
    /// as many as a sized array's size, or any number for an unsized one.
    Array(Vec<ResourceKind>),
}

/// What a job gave.
#[derive(Debug, Default, Serialize)]
pub struct Reply {
    /// The job's id.
    pub id: serde_json::Value,
    /// Why the job did not run, where it did not.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Per invocation of a vertex or pixel program, in the job's order.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub invocations: Vec<Invocation>,
    /// The writable resources (storage buffers and storage textures) after the run: a buffer's words; a storage
    /// texture's texels of every layer, row by row, a word per lane its format has.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub resources: Vec<Written>,
    /// The adapter it ran on.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub adapter: String,
}

/// What one invocation gave.
#[derive(Debug, Serialize)]
pub struct Invocation {
    /// Whether it returned: a discarded pixel's results are not written.
    pub returned: bool,
    /// The entry point's results, each member of its result structure in order (or its one result), as an argument's
    /// words are given; empty where it did not return.
    pub results: Vec<Vec<u32>>,
}

/// A writable resource after the run.
#[derive(Debug, Serialize)]
pub struct Written {
    /// Its bind group.
    pub group: u32,
    /// Its binding.
    pub binding: u32,
    /// Its element, where it is an element of a binding array.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub element: Option<u32>,
    /// Its words.
    pub words: Vec<u32>,
}
