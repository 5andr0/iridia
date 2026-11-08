//! The WGSL execution harness (`tools.wgsl_run`, built from tools/wgsl-run): one process, jobs run in turn. A WGSL
//! program's entry point runs on Direct3D 12's WARP adapter through wgpu, an executor iridia does not write, to hold
//! what this crate writes to the IR executor's definition of the same program.
#![allow(dead_code, reason = "each test uses the parts it needs")]

use std::io::{BufRead as _, BufReader, Write as _};
use std::process::{Child, ChildStdin, Stdio};
use std::sync::{Mutex, MutexGuard, OnceLock, mpsc};
use std::time::Duration;

use crate::inputs::{self, Tool};

/// How long one job may take, compiles included.
const LIMIT: Duration = Duration::from_secs(120);

pub struct Warp {
    /// The cross-process lock on the adapter (`primase_testdata::warp_exclusive`), held for as long as the harness lives.
    _exclusive: primase_testdata::WarpLock,
    child: Child,
    stdin: ChildStdin,
    replies: mpsc::Receiver<String>,
}

impl Drop for Warp {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Warp {
    /// The configured harness, shared by every test of the binary (they take turns); `None` (the test skips its WARP
    /// half) where `tools.wgsl_run` is not set. Its process lives until the test binary exits.
    pub fn shared() -> Option<MutexGuard<'static, Warp>> {
        static SHARED: OnceLock<Option<Mutex<Warp>>> = OnceLock::new();
        SHARED
            .get_or_init(|| Warp::start().map(Mutex::new))
            .as_ref()
            .map(|m| m.lock().unwrap_or_else(std::sync::PoisonError::into_inner))
    }

    fn start() -> Option<Warp> {
        let tool = inputs::program(Tool::WgslRun)?;
        let exclusive = primase_testdata::warp_exclusive();
        let mut command = tool.command();
        if let Some(dxc) = inputs::tool(Tool::WgslRunDxc) {
            command.arg("--dxc").path_arg(dxc);
        }
        command.args(["--timeout", "30"]);
        let mut child = command
            .command()
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap_or_else(|e| panic!("starting {}: {e}", tool.path().display()));
        let stdin = child.stdin.take().expect("standard input");
        let stdout = child.stdout.take().expect("standard output");
        let (send, replies) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if send.send(line).is_err() {
                    break;
                }
            }
        });
        Some(Warp {
            _exclusive: exclusive,
            child,
            stdin,
            replies,
        })
    }

    /// Run one job: its reply, or the harness's error.
    pub fn run(&mut self, job: &serde_json::Value) -> Result<serde_json::Value, String> {
        let line = serde_json::to_string(job).map_err(|e| e.to_string())?;
        writeln!(self.stdin, "{line}")
            .and_then(|()| self.stdin.flush())
            .map_err(|e| format!("writing a job: {e}"))?;
        let reply = self
            .replies
            .recv_timeout(LIMIT)
            .map_err(|_| format!("no reply within {} s", LIMIT.as_secs()))?;
        let v: serde_json::Value =
            serde_json::from_str(&reply).map_err(|e| format!("the reply {reply:?}: {e}"))?;
        match v.get("error").and_then(|e| e.as_str()) {
            Some(e) => Err(e.to_owned()),
            None => Ok(v),
        }
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
