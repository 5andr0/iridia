"""Multisampled textures through the harness: each class (float, unsigned, signed, depth) at 1, 2, 4, 8 and 16 samples, each
sample given its own texel and read back by a compute program with textureLoad and textureNumSamples.

python probes/msaa.py <wgsl-run.exe> <dxcompiler.dll>"""
import json
import struct
import subprocess
import sys

exe, dxc = sys.argv[1:3]


def bits(f):
    return struct.unpack('<I', struct.pack('<f', f))[0]


def program(kind):
    ty = {'f': 'f32', 'u': 'u32', 'i': 'i32'}[kind]
    conv = {'f': 'bitcast<u32>(v.x)', 'u': 'v.x', 'i': 'bitcast<u32>(v.x)'}[kind]
    return f"""@group(0) @binding(0) var t: texture_multisampled_2d<{ty}>;
@group(0) @binding(1) var<storage, read_write> o: array<u32>;
@compute @workgroup_size(1)
fn main() {{
    let n = textureNumSamples(t);
    o[0] = n;
    for (var i = 0u; i < n; i++) {{
        let v = textureLoad(t, vec2<i32>(1, 2), i32(i));
        o[1u + i] = {conv};
    }}
}}
"""


depth = """@group(0) @binding(0) var t: texture_depth_multisampled_2d;
@group(0) @binding(1) var<storage, read_write> o: array<u32>;
@compute @workgroup_size(1)
fn main() {
    let n = textureNumSamples(t);
    o[0] = n;
    for (var i = 0u; i < n; i++) {
        o[1u + i] = bitcast<u32>(textureLoad(t, vec2<i32>(1, 2), i32(i)));
    }
}
"""

jobs = []
for count in (1, 2, 4, 8, 16):
    per = [[bits(0.125 * (k + 1)), 0, 0, bits(1.0)] for k in range(count)]
    jobs.append((f'f{count}', program('f'), count, per, [bits(0.125 * (k + 1)) for k in range(count)]))
    per_u = [[100 + k, 0, 0, 1] for k in range(count)]
    jobs.append((f'u{count}', program('u'), count, per_u, [100 + k for k in range(count)]))
    per_i = [[(-5 - k) & 0xffffffff, 0, 0, 1] for k in range(count)]
    jobs.append((f'i{count}', program('i'), count, per_i, [(-5 - k) & 0xffffffff for k in range(count)]))
    per_d = [[bits(0.0625 * (k + 1)), 0, 0, 0] for k in range(count)]
    jobs.append((f'd{count}', depth, count, per_d, [bits(0.0625 * (k + 1)) for k in range(count)]))

p = subprocess.Popen([exe, '--dxc', dxc, '--timeout', '30'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
for name, wgsl, count, per, want in jobs:
    job = {'id': name, 'wgsl': wgsl, 'dispatch': [1, 1, 1], 'resources': [
        {'group': 0, 'binding': 0, 'texture': {'texel': [0, 0, 0, 0], 'samples': count, 'per_sample': per, 'size': [4, 4, 1]}},
        {'group': 0, 'binding': 1, 'buffer': {'words': [0] * 20}},
    ]}
    p.stdin.write(json.dumps(job) + '\n')
    p.stdin.flush()
    r = json.loads(p.stdout.readline())
    if 'error' in r:
        print(name, 'ERROR', r['error'][:200])
        continue
    words = r['resources'][0]['words']
    ok = words[0] == count and words[1:1 + count] == want
    print(name, 'ok' if ok else f'DIFFER {words[:1 + count]} want {count} {want}')
p.stdin.close()
