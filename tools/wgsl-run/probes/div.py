"""Measure WARP's `a / b` through WGSL against the correctly rounded f32 quotient, and what `fract(saturate(a / b) * 65025)`
makes of it: the measurement behind iridia-dxbc/tests/exec_wgsl.rs's note on amplified rounding of a division (Elden
Ring's ambient occlusion packs depth that way).

usage: python div.py <wgsl-run.exe> <dxcompiler.dll>
"""
import json
import subprocess
import sys

import numpy as np

N = 4096
rng = np.random.default_rng(20261108)
a = rng.uniform(0.0, 1.5, N).astype(np.float32)
b = rng.uniform(0.5, 2.0, N).astype(np.float32)
wgsl = """@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    out[g.x] = bitcast<u32>(a[g.x] / b[g.x]);
}
"""
words = lambda x: [int(w) for w in x.view(np.uint32)]
job = {"id": "div", "wgsl": wgsl, "dispatch": [N // 64, 1, 1], "resources": [
    {"group": 0, "binding": 0, "buffer": {"words": words(a)}},
    {"group": 0, "binding": 1, "buffer": {"words": words(b)}},
    {"group": 0, "binding": 2, "buffer": {"words": [0] * N}},
]}
r = subprocess.run([sys.argv[1], '--dxc', sys.argv[2]], input=json.dumps(job) + '\n', capture_output=True, text=True)
reply = json.loads(r.stdout.splitlines()[0])
if 'error' in reply:
    sys.exit(reply['error'])
warp = np.array(reply['resources'][0]['words'], dtype=np.uint32).view(np.float32)
exact = a / b  # numpy divides f32 by f32 as IEEE 754 does: correctly rounded
ulps = np.abs(warp.view(np.int32).astype(np.int64) - exact.view(np.int32).astype(np.int64))
print(f"{N} quotients a / b, a in 0..1.5, b in 0.5..2")
print(f"WARP vs correctly rounded: exact {int((ulps == 0).sum())}, 1 ulp {int((ulps == 1).sum())}, "
      f"2 ulp {int((ulps == 2).sum())}, more {int((ulps > 2).sum())}, max {int(ulps.max())} ulp")
sat = lambda x: np.clip(x, 0, 1).astype(np.float32)
pack = lambda x: np.mod(sat(x) * np.float32(65025), np.float32(1.0)).astype(np.float64)
apart = np.abs(pack(warp) - pack(exact))
apart = np.minimum(apart, 1 - apart)
print(f"fract(saturate(a / b) * 65025) apart by more than 1e-3: {int((apart > 1e-3).sum())} of {N}")
