"""iridia_sin (iridia-dxbc/tests/wgsl_run/reference_sin.wgsl, what `shader.sin=reference` swaps in) and WARP's builtin sin on WARP,
against the f64 reference rounded to f32, over random arguments in -1000..1000 and -7..7 and the cell arguments of Unity
program 04211: how many are the correctly rounded value, the worst ulps and the worst absolute error of each.
usage: sin_ref.py <wgsl-run.exe> <dxcompiler.dll> [n]"""
import json, math, random, struct, subprocess, sys, os
lib = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'iridia-dxbc', 'tests', 'wgsl_run', 'reference_sin.wgsl')).read()
exe, dxc = sys.argv[1:3]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
f32 = lambda x: struct.unpack('<f', struct.pack('<f', x))[0]
bits = lambda x: struct.unpack('<I', struct.pack('<f', x))[0]
frombits = lambda w: struct.unpack('<f', struct.pack('<I', w))[0]
random.seed(7)
args = [f32(random.uniform(-1000, 1000)) for _ in range(n // 2)]
args += [f32(cx * f32(127.1) + cy * f32(311.7)) for cx in range(-4, 5) for cy in range(-4, 5)]
args += [f32(cx * f32(269.5) + cy * f32(183.3)) for cx in range(-4, 5) for cy in range(-4, 5)]
args += [f32(random.uniform(-7, 7)) for _ in range(n // 2)]
src = lib + """
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) w: vec3<u32>) {
    out[w.x] = bitcast<u32>(SIN(a[w.x]));
}
"""
args = args[:65535]
m = len(args)


def run(fn):
    job = {"id": "s", "wgsl": src.replace("SIN", fn), "dispatch": [m, 1, 1], "resources": [
        {"group": 0, "binding": 0, "buffer": {"words": [bits(x) for x in args]}},
        {"group": 0, "binding": 1, "buffer": {"words": [0] * m}}]}
    r = subprocess.run([exe, '--dxc', dxc], input=json.dumps(job) + chr(10), capture_output=True, text=True)
    reply = json.loads(r.stdout.splitlines()[0])
    if 'error' in reply:
        sys.exit(reply['error'][:3000])
    return reply['resources'][0]['words']


for fn in ('iridia_sin', 'sin'):
    got = run(fn)
    exact = off1 = worst = 0
    for x, g in zip(args, got):
        ref = bits(f32(math.sin(x)))
        d = abs(ref - g) if (ref >> 31) == (g >> 31) else 10**9
        exact += d == 0; off1 += d == 1; worst = max(worst, d)
    err = max(abs(math.sin(x) - frombits(g)) for x, g in zip(args, got))
    print(f"{fn:11} {m} arguments: {exact} the correctly rounded sin, {off1} one ulp off, worst {worst} ulp, max abs error {err:.3e}")
