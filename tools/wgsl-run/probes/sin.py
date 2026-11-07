"""Measure WARP's sin at hash-sized arguments against correctly rounded f32 sin, and what fract(sin * 43758.5453) makes
of it: the measurement behind iridia-dxbc/tests/exec_wgsl.rs's note on amplified rounding.

usage: python sin.py <wgsl-run.exe> <dxcompiler.dll>
"""
import json
import math
import struct
import subprocess
import sys

N = 64
f32 = lambda x: struct.unpack('<f', struct.pack('<f', x))[0]
bits = lambda x: struct.unpack('<I', struct.pack('<f', x))[0]
frombits = lambda w: struct.unpack('<f', struct.pack('<I', w))[0]

# arguments like dot(p, (127.1, 311.7)) for p of a few units to a few hundred
args = [f32(127.1 * (1.5 + 3.7 * i) + 311.7 * (0.25 + 1.3 * i)) for i in range(N)]
wgsl = """@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) w: vec3<u32>) {
    out[w.x] = bitcast<u32>(sin(a[w.x]));
}
"""
job = {"id": "sin", "wgsl": wgsl, "dispatch": [N, 1, 1], "resources": [
    {"group": 0, "binding": 0, "buffer": {"words": [bits(x) for x in args]}},
    {"group": 0, "binding": 1, "buffer": {"words": [0] * N}},
]}
r = subprocess.run([sys.argv[1], '--dxc', sys.argv[2]], input=json.dumps(job) + '\n', capture_output=True, text=True)
reply = json.loads(r.stdout.splitlines()[0])
if 'error' in reply:
    sys.exit(reply['error'])
warp = [frombits(w) for w in reply['resources'][0]['words']]

ulps = []
hash_apart = 0
for x, w in zip(args, warp):
    exact = f32(math.sin(x))
    d = abs(bits(exact) - bits(w)) if (exact >= 0) == (w >= 0) else None
    ulps.append(d)
    h_exact = (f32(exact * f32(43758.546875))) % 1.0
    h_warp = (f32(w * f32(43758.546875))) % 1.0
    if abs(h_exact - h_warp) > 1e-3:
        hash_apart += 1
known = [u for u in ulps if u is not None]
print(f"{N} arguments from {min(args):.0f} to {max(args):.0f}")
print(f"WARP sin vs correctly rounded f32 sin: exact {known.count(0)}, max {max(known)} ulp, "
      f"median {sorted(known)[len(known) // 2]} ulp, sign differs {ulps.count(None)}")
print(f"fract(sin * 43758.546875) apart by more than 1e-3: {hash_apart} of {N}")
for x, w in list(zip(args, warp))[:6]:
    print(f"  sin({x}) WARP {w!r} exact {f32(math.sin(x))!r}")
