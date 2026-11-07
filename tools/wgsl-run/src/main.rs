//! `wgsl-run`: runs WGSL programs on Direct3D 12's WARP adapter through wgpu, for iridia's tests. Jobs arrive one JSON
//! object a line on standard input and each gets one JSON reply a line on standard output (README.md states the
//! format). The device is made once; a job whose GPU work does not finish within the timeout is answered with an error
//! and the process exits with status 2, since the device is still busy (the caller starts another).
//!
//! `wgsl-run [--dxc <dxcompiler.dll>] [--timeout <seconds>]`; `wgsl-run --adapter` prints the adapter and exits.

mod gpu;
mod job;
mod wrap;

use std::io::{BufRead as _, Write as _};
use std::panic::AssertUnwindSafe;
use std::time::Duration;

use gpu::{Failure, Gpu};
use job::{Job, Reply};

fn main() {
    let mut dxc = None;
    let mut timeout = Duration::from_secs(20);
    let mut adapter_only = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--dxc" => dxc = args.next(),
            "--timeout" => {
                let s = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| usage());
                timeout = Duration::from_secs(s);
            }
            "--adapter" => adapter_only = true,
            _ => usage(),
        }
    }
    let make = || Gpu::new(dxc.clone(), timeout).unwrap_or_else(|e| fail(&e));
    let mut gpu = make();
    if adapter_only {
        println!("{}", gpu.adapter);
        return;
    }
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line.unwrap_or_else(|e| fail(&format!("standard input: {e}")));
        if line.trim().is_empty() {
            continue;
        }
        let (reply, exit) = match serde_json::from_str::<Job>(&line) {
            Err(e) => (error(serde_json::Value::Null, format!("the job: {e}")), false),
            Ok(job) => match std::panic::catch_unwind(AssertUnwindSafe(|| gpu.run(&job))) {
                Ok(Ok(mut reply)) => {
                    reply.id = job.id;
                    (reply, false)
                }
                Ok(Err(Failure::Error(e))) => (error(job.id, e), false),
                Ok(Err(Failure::TimedOut(e))) => (error(job.id, e), true),
                Err(panic) => {
                    let why = panic
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                        .unwrap_or_default();
                    // the device may be in any state after a panic: a new one for the next job
                    gpu = make();
                    (error(job.id, format!("the harness panicked: {why}")), false)
                }
            },
        };
        let text = serde_json::to_string(&reply).unwrap_or_else(|e| fail(&format!("the reply: {e}")));
        writeln!(out, "{text}")
            .and_then(|()| out.flush())
            .unwrap_or_else(|e| fail(&format!("standard output: {e}")));
        if exit {
            std::process::exit(2);
        }
    }
}

fn error(id: serde_json::Value, e: String) -> Reply {
    Reply {
        id,
        error: Some(e),
        ..Reply::default()
    }
}

fn usage() -> ! {
    fail("usage: wgsl-run [--dxc <dxcompiler.dll>] [--timeout <seconds>] [--adapter]")
}

fn fail(e: &str) -> ! {
    eprintln!("wgsl-run: {e}");
    std::process::exit(1)
}
