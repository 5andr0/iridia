"""WARP's sin of the arguments SpeedTree_50777_w32082_Animation takes at vector 0, against an f64 reference.

The program unpacks the halves of the words of t30 and takes fract(abs(sin(x) * 100000)) of the low half of word 0 and the
low half of word 1 (the arguments printed below are the low and the high half of the first words, the lanes
`shader.lanes` prints as "first words"). The executor's sin is `narrow(f64::from(x).sin())` (iridia-exec-ir's FSin),
which is the f64 reference rounded to f32: what is measured here is WARP's value against it.

D3D's bound, for reading the table: Microsoft Learn's `sincos (sm4 - asm)` page says "The maximum absolute error is 0.0008 in the
interval from -100*Pi to +100*Pi" and promises nothing beyond (it is the SM4 instruction's page; that the DXIL path WARP
runs here keeps it is not checked).

The log is the output of the corpus test with the program's name in `shader.lanes`:
  GAME_DATA_DEBUG="shader.compute=only;shader.only=SpeedTree_50777_w32082_;shader.lanes=SpeedTree_50777_w32082_"     cargo test --release -p iridia-dxbc --test exec_wgsl -- --nocapture elden_programs > log.txt

usage: python sin_args.py <wgsl-run.exe> <dxcompiler.dll> <log>
"""
import json
import math
import re
import struct
import subprocess
import sys

exe, dxc, log = sys.argv[1:4]
f32 = lambda x: struct.unpack('<f', struct.pack('<f', x))[0]
bits = lambda x: struct.unpack('<I', struct.pack('<f', x))[0]
frombits = lambda w: struct.unpack('<f', struct.pack('<I', w))[0]


def half(h):
    return struct.unpack('<e', struct.pack('<H', h))[0]


text = open(log, encoding='utf-8', errors='replace').read()
m = re.search(r'case 0 resource (\d+) \(group 1 binding 30\) first words \[([0-9a-f, ]+)\]', text)
if not m:
    sys.exit('no "case 0 resource N (group 1 binding 30) first words" line in the log')
words = [int(w, 16) for w in m.group(2).split(',')]
args = []
for w in words[:8]:
    args += [('lo', w, half(w & 0xFFFF)), ('hi', w, half(w >> 16))]

wgsl = """@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) w: vec3<u32>) {
    out[w.x] = bitcast<u32>(sin(a[w.x]));
}
"""
n = len(args)
job = {"id": "sin", "wgsl": wgsl, "dispatch": [n, 1, 1], "resources": [
    {"group": 0, "binding": 0, "buffer": {"words": [bits(x) for _, _, x in args]}},
    {"group": 0, "binding": 1, "buffer": {"words": [0] * n}},
]}
r = subprocess.run([exe, '--dxc', dxc], input=json.dumps(job) + '\n', capture_output=True, text=True)
reply = json.loads(r.stdout.splitlines()[0])
if 'error' in reply:
    sys.exit(reply['error'])
warp = reply['resources'][0]['words']
print("D3D's bound (Microsoft Learn, sincos sm4 asm): maximum absolute error 0.0008 in [-100*Pi, 100*Pi]; no bound outside")
print(f"{'half':>4} {'word':>10} {'x':>12} {'in range':>8} {'executor sin(x)':>17} {'WARP sin(x)':>14} {'abs error':>10} {'ulp apart':>10} {'x1e5 fract: executor':>22} {'WARP':>10}")
for (which, w, x), got in zip(args, warp):
    ref = f32(math.sin(x))
    d = abs(bits(ref) - got) if (bits(ref) >> 31) == (got >> 31) else 'sign'
    hr = f32(abs(f32(ref * 100000.0))) % 1.0
    hw = f32(abs(f32(frombits(got) * 100000.0))) % 1.0
    inside = abs(x) <= 100 * math.pi
    err = abs(ref - frombits(got))
    print(f"{which:>4} {w:#010x} {x:12.5g} {'yes' if inside else 'NO':>8} {ref:17.9g} {frombits(got):14.9g} {err:10.2e} {d:>10} {hr:22.6f} {hw:10.6f}")
