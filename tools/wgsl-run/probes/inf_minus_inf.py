"""`-x + x` and `x - x` of an infinity on WARP (DXC, then FXC): iridia-exec-ir gives NaN (IEEE: inf - inf), and one vkd3d rcp
program of the d3d9 corpus (`-r + r` of `r = 1/0`, then a compare) answers as if it were 0. Which of the compiler and WARP does it:
each form is run on a uniform 0 (so nothing folds at compile time), its result bits printed.
usage: inf_minus_inf.py <wgsl-run.exe> <dxcompiler.dll>"""
import json, struct, subprocess, sys
exe, dxc = sys.argv[1:3]
bits = lambda x: struct.unpack('<I', struct.pack('<f', x))[0]
forms = {
    "-r + r": "-r + r",
    "r - r": "r - r",
    "r + (-r)": "r + (-r)",
    "r + bitcast<f32>(bitcast<u32>(r) ^ 0x80000000u)": "r + bitcast<f32>(bitcast<u32>(r) ^ 0x80000000u)",
    "r * 0 (inf * 0)": "r * z",
    "r / r": "r / r",
    "r (the infinity)": "r",
}
body = "\n".join(f"    out[{i}] = bitcast<u32>({e});" for i, e in enumerate(forms.values()))
src = f"""
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1) fn main() {{
    let r = 1.0f / a[0];
    let z = a[0];
{body}
}}
"""
for use_dxc in (True, False):
    job = {"id": "p", "wgsl": src, "dispatch": [1, 1, 1], "resources": [
        {"group": 0, "binding": 0, "buffer": {"words": [bits(0.0)]}},
        {"group": 0, "binding": 1, "buffer": {"words": [0] * len(forms)}}]}
    cmd = [exe] + (['--dxc', dxc] if use_dxc else [])
    r = subprocess.run(cmd, input=json.dumps(job) + chr(10), capture_output=True, text=True)
    reply = json.loads(r.stdout.splitlines()[0])
    if 'error' in reply:
        print("DXC" if use_dxc else "FXC", "error:", reply['error'][:300])
        continue
    words = reply['resources'][0]['words']
    print("DXC" if use_dxc else "FXC")
    for (name, _), w in zip(forms.items(), words):
        print(f"  {name:50} {w:#010x} {struct.unpack('<f', struct.pack('<I', w))[0]}")
