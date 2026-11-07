"""Measure what WARP, through WGSL, makes of `1.0 - (a * 0.85 + 0.15)`: which of three f32 evaluations it equals, lane by
lane: the unfused one (a multiply, an add, a subtract: what Direct3D's `mad` and `add` give the executor, and WARP running
the DXBC), the fused one (`fma(a, 0.85, 0.15)`, then the subtract), and the one DXC's fast-math folding gives
(`(1 - 0.15) - a * 0.85`, the constants combined). The measurement behind iridia-dxbc/tests/exec_wgsl.rs's note on
"near zero" lanes.

usage: python mad.py <wgsl-run.exe> <dxcompiler.dll>
"""
import json
import subprocess
import sys
from fractions import Fraction

import numpy as np

N = 4096
rng = np.random.default_rng(20261108)
a = rng.uniform(-0.25, 1.75, N).astype(np.float32)
wgsl = """@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    out[g.x] = bitcast<u32>(1.0 - (a[g.x] * 0.85 + 0.15));
}
"""
job = {"id": "mad", "wgsl": wgsl, "dispatch": [N // 64, 1, 1], "resources": [
    {"group": 0, "binding": 0, "buffer": {"words": [int(w) for w in a.view(np.uint32)]}},
    {"group": 0, "binding": 1, "buffer": {"words": [0] * N}},
]}
r = subprocess.run([sys.argv[1], '--dxc', sys.argv[2]], input=json.dumps(job) + '\n', capture_output=True, text=True)
reply = json.loads(r.stdout.splitlines()[0])
if 'error' in reply:
    sys.exit(reply['error'])
warp = np.array(reply['resources'][0]['words'], dtype=np.uint32)

c, d, one = np.float32(0.85), np.float32(0.15), np.float32(1)
unfused = np.array([np.float32(one - np.float32(np.float32(x * c) + d)) for x in a]).view(np.uint32)
fused = np.array([
    np.float32(one - np.float32(float(Fraction(float(x)) * Fraction(float(c)) + Fraction(float(d))))) for x in a
]).view(np.uint32)
folded = np.array([np.float32(np.float32(one - d) - np.float32(x * c)) for x in a]).view(np.uint32)
print(f"{N} values of a in -0.25..1.75")
print(f"WARP equals the unfused evaluation in {int((warp == unfused).sum())}, the fused in {int((warp == fused).sum())}, "
      f"the folded (1 - 0.15) - a * 0.85 in {int((warp == folded).sum())}")
