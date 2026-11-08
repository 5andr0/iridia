"""WARP's sin of the arguments Unity 04211's cell hash `fract(sin(dot(cell, k)) * 43758.547)` takes for integer cells -3..3 in
both coordinates (k = (127.1, 311.7) and (269.5, 183.3)), against the f64 reference rounded to f32 (the executor's sin).
usage: python sin_cells.py <wgsl-run.exe> <dxcompiler.dll>"""
import json, math, struct, subprocess, sys
exe, dxc = sys.argv[1:3]
f32 = lambda x: struct.unpack('<f', struct.pack('<f', x))[0]
bits = lambda x: struct.unpack('<I', struct.pack('<f', x))[0]
frombits = lambda w: struct.unpack('<f', struct.pack('<I', w))[0]
args = []
for k in ((127.1, 311.7), (269.5, 183.3)):
    kx, ky = f32(k[0]), f32(k[1])
    for cx in range(-3, 4):
        for cy in range(-3, 4):
            args.append(f32(f32(cx * kx) + f32(cy * ky)))
wgsl = """@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) w: vec3<u32>) {
    out[w.x] = bitcast<u32>(sin(a[w.x]));
}
"""
n = len(args)
job = {"id": "s", "wgsl": wgsl, "dispatch": [n, 1, 1], "resources": [
    {"group": 0, "binding": 0, "buffer": {"words": [bits(x) for x in args]}},
    {"group": 0, "binding": 1, "buffer": {"words": [0] * n}}]}
r = subprocess.run([exe, '--dxc', dxc], input=json.dumps(job) + '\n', capture_output=True, text=True)
reply = json.loads(r.stdout.splitlines()[0])
if 'error' in reply:
    sys.exit(reply['error'])
warp = reply['resources'][0]['words']
fr = lambda v: v - math.floor(v)
bad = 0; worst = 0; maxerr = 0; outside = 0
for x, got in zip(args, warp):
    ref = f32(math.sin(x))
    w = frombits(got)
    maxerr = max(maxerr, abs(ref - w))
    if abs(x) > 100 * math.pi: outside += 1
    dh = abs(fr(f32(abs(ref) * 43758.547 if False else ref * 43758.547)) - fr(f32(w * 43758.547)))
    dh = min(dh, 1 - dh)
    worst = max(worst, dh)
    bad += dh > 1e-3
print(f"{n} arguments, {outside} outside D3D's -100pi..100pi bound; max |WARP sin - reference| = {maxerr:.3e}")
print(f"hash fract(sin*43758.547) apart by more than 1e-3 for {bad} of {n}; worst {worst:.4f}")
