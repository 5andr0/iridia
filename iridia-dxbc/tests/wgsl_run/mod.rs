//! The WGSL harness the tests run (`iridia-warp-client`, the client of tools/wgsl-run), found by this crate's configuration
//! (`tests/inputs`): everything the client has, and the harness and Tint it makes of the configured tools.
#![allow(dead_code, reason = "each test uses the parts it needs")]

use std::path::PathBuf;

pub use iridia_warp_client::*;

use crate::inputs::{self, Tool};

/// The configured harness; `None` (the test skips) where `tools.wgsl_run` is not set. `tools.wgsl_run_dxc`, the
/// `dxcompiler.dll` DXC compiles with (the programs that need Shader Model 6: subgroup operations, 16-bit and 64-bit
/// types), is used where it is configured, FXC otherwise.
pub fn harness() -> Option<Harness> {
    Some(Harness::new(
        inputs::program(Tool::WgslRun)?,
        inputs::tool(Tool::WgslRunDxc),
    ))
}

/// Dawn's `tint` (`tools.tint`); `None` where it is not set. `shader.tint_keep` is a folder that keeps the text Tint refuses
/// and what it said.
pub fn tint() -> Option<Tint> {
    Some(Tint::new(
        inputs::program(Tool::Tint)?,
        inputs::debug("shader.tint_keep").map(PathBuf::from),
    ))
}
