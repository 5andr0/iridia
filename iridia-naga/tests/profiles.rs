//! What each profile binds with: the browser's plan keeps inside WebGPU's default limits, naga's has Direct3D's
//! registers as they are (the plans are `iridia_ir::target::BindingPlan`'s, tested there).

use iridia_naga::{Bindings, Profile};

#[test]
fn the_browser_profile_binds_inside_webgpu_limits() {
    assert_eq!(Profile::BROWSER.bindings, Bindings::WebGpu);
    assert_eq!(Profile::NAGA.bindings, Bindings::D3d);
}
