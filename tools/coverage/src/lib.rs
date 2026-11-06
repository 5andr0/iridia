//! What iridia and dx read, list, lift, run and write, per construct, measured by running them over test suites.
//!
//! - [`vkd3d`]: vkd3d's HLSL shader tests, read as runs of its Windows runners and run on a [`device::Device`].
//! - [`offload`]: LLVM's offload-test-suite, read as lit runs its tests for DirectX with DXC, and run on the device.
//! - [`table`]: the coverage table made of the suites' runs.
//! - [`front`]: the one place the runners learn how iridia's public pipeline reads, lifts and writes a program.
#![allow(
    clippy::cast_possible_truncation,
    reason = "texel and word arithmetic on sizes a test file bounds"
)]

pub mod device;
pub mod front;
pub mod inflate;
pub mod offload;
pub mod stages;
pub mod table;
pub mod vkd3d;
