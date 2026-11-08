//! The WGSL execution harness (`tools.wgsl_run`, built from tools/wgsl-run): a WGSL program's entry point runs on
//! Direct3D 12's WARP adapter through wgpu, an executor iridia does not write, to hold what this crate writes to the IR
//! executor's definition of the same program. The client is `iridia-warp-client`'s, which takes the machine-wide lock on
//! the adapter for each job.
#![allow(dead_code, reason = "each test uses the parts it needs")]

use std::sync::{Mutex, MutexGuard, OnceLock};

use iridia_warp_client::Harness;

use crate::inputs::{self, Tool};

pub struct Warp {
    harness: Harness,
}

impl Warp {
    /// The configured harness, shared by every test of the binary (they take turns); `None` (the test skips its WARP
    /// half) where `tools.wgsl_run` is not set. Its process lives until the test binary exits.
    pub fn shared() -> Option<MutexGuard<'static, Warp>> {
        static SHARED: OnceLock<Option<Mutex<Warp>>> = OnceLock::new();
        SHARED
            .get_or_init(|| {
                Some(Mutex::new(Warp {
                    harness: Harness::new(inputs::program(Tool::WgslRun)?, inputs::tool(Tool::WgslRunDxc)),
                }))
            })
            .as_ref()
            .map(|m| m.lock().unwrap_or_else(std::sync::PoisonError::into_inner))
    }

    /// Run one job: its reply, or the harness's error.
    pub fn run(&mut self, job: &serde_json::Value) -> Result<serde_json::Value, String> {
        self.harness.run(job)
    }

    /// A pixel program's result target 0 for each invocation's arguments, four lanes of bits (zero past the lanes the
    /// output has): `None` for an invocation that did not return.
    pub fn pixel(&mut self, wgsl: &str, given: &[Vec<[u32; 4]>]) -> Result<Vec<Option<[u32; 4]>>, String> {
        self.pixel_with(wgsl, given, &serde_json::json!([]))
    }

    /// [`Warp::pixel`] with the job's `resources`.
    pub fn pixel_with(
        &mut self,
        wgsl: &str,
        given: &[Vec<[u32; 4]>],
        resources: &serde_json::Value,
    ) -> Result<Vec<Option<[u32; 4]>>, String> {
        let reply = self
            .run(&serde_json::json!({"id": 0, "wgsl": wgsl, "invocations": given, "resources": resources}))?;
        let got = reply["invocations"]
            .as_array()
            .ok_or("no invocations in the reply")?;
        if got.len() != given.len() {
            return Err(format!("{} invocations answered {}", given.len(), got.len()));
        }
        got.iter()
            .map(|i| {
                if i["returned"] != true {
                    return Ok(None);
                }
                let mut lanes = [0u32; 4];
                for (k, lane) in lanes.iter_mut().enumerate() {
                    *lane =
                        u32::try_from(i["results"][0][k].as_u64().unwrap_or(0)).map_err(|e| e.to_string())?;
                }
                Ok(Some(lanes))
            })
            .collect()
    }
}
