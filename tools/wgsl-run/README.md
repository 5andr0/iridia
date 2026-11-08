# wgsl-run

Runs a WGSL program's entry point on Direct3D 12's WARP adapter through [wgpu](https://github.com/gfx-rs/wgpu) 30.0.1, for iridia's tests: an executor iridia does not write (wgpu's naga reads the WGSL and writes HLSL, DXC or FXC compiles it, WARP runs it), to hold a writer's WGSL to iridia's own IR executor. Windows only. It is its own Cargo workspace, so wgpu stays out of iridia's lockfile; it is not published.

```
cargo build --release
wgsl-run [--dxc <path to dxcompiler.dll>] [--timeout <seconds>]
wgsl-run --adapter
```

`--dxc` compiles with DXC: `dxcompiler.dll` of [DXC](https://github.com/microsoft/DirectXShaderCompiler/releases) 1.8.2502 or later (wgpu 30's minimum), `dxil.dll` beside it. The Windows SDK's (10.0.26100's, 1.7.2308) is older: every program it compiles fails with "DXIL signing library (dxil.dll) not found". Without it FXC compiles, which has no Shader Model 6 (no subgroup operations, no 16-bit or 64-bit types). `--timeout` (20 s by default) bounds a job's GPU work. `--adapter` prints the adapter and the device's features and exits. Tests find it as `tools.wgsl_run` in the game-data configuration, and the `dxcompiler.dll` to pass as `--dxc` as `tools.wgsl_run_dxc` (without it they compile with FXC, and the programs that need Shader Model 6 fail).

## How a program runs

The WGSL is read by wgpu's naga and validated. A vertex or pixel program's entry point becomes a plain function, and the harness adds an entry point that reads each invocation's arguments from a storage buffer, calls it, and writes its results to another, then a word saying it returned (both buffers are in bind group 7, so a program may use groups 0 to 6):

- a vertex program is called from a compute entry point, one workgroup an invocation: no rasterizer, so no interpolation touches its inputs, and its built-ins (`position`, `clip_distances`) are plain results;
- a pixel program is called from a fragment entry point, one pixel an invocation, over 2x2 quads of a target two pixels high: its derivatives are taken between the quad's lanes by WARP itself, and an invocation that discards writes nothing after it, so its results are reported as not returned. Every argument, built-ins included (`position`, `front_facing`), is the job's, bit for bit.

A compute program runs as it is, dispatched over the job's workgroups. Every resource the entry point uses must be in the job; each writable one (a storage buffer, a storage texture) is read back after the run.

## Jobs

One JSON object a line on standard input; one reply a line on standard output, in order. The device is made once. A job whose GPU work does not finish within the timeout is answered with an error, and the process exits with status 2 (the device is still busy): the caller starts another.

```json
{
  "id": "anything, echoed",
  "wgsl": "…",
  "entry": "main",
  "invocations": [ [[1065353216, 0, 0, 1065353216], [7]], … ],
  "dispatch": [2, 1, 1],
  "resources": [
    { "group": 0, "binding": 0, "buffer": { "words": [0, 0, 0, 0, 1073741824, 0, 0, 0] } },
    { "group": 1, "binding": 0, "texture": { "texel": [1040187392, 1048576000, 1056964608, 1065353216] } },
    { "group": 2, "binding": 0, "sampler": {} }
  ]
}
```

- `entry`: the entry point; the module's only one where absent.
- `invocations` (vertex and pixel programs): each invocation's entry arguments in order, each argument's 32-bit words: a scalar one word, a vector one a lane, a bool 0 or 1, a 64-bit lane two (low first), a structure its members' words in order. A pixel program's invocations run four at a time as a quad: invocation `4q + k` is lane `k` of quad `q` (0 top left, 1 top right, 2 bottom left, 3 bottom right), so their count must be a multiple of 4.
- `dispatch` (compute programs): the workgroups along x, y and z.
- `resources`: by `group` and `binding`, one of
  - `buffer`: a uniform or storage buffer's words. A uniform buffer is as long as its type (the words past those given are zero); a storage buffer is as long as the words given.
  - `texture`: a sampled, depth or storage texture whose every texel of every layer is `texel` (four lanes' bits; a depth texture's depth, which must be within 0 to 1, and an `r32` storage texture's value are the first lane). 4 texels a side and 4 deep (`size`: `[w, h, d]` to change it), one mip level, 2 layers for an array, 6 for a cube, 12 for a cube array (`layers` to change it). Sampled float textures are `rgba32float`, integer ones `rgba32uint` and `rgba32sint`, depth ones `depth32float`; a storage texture has its declared format. A multisampled texture is a `texture_multisampled_2d<f32>`, `<i32>` or `<u32>` or a `texture_depth_multisampled_2d` (not an array, not 1D, 3D or a cube): `samples` of it (4 where absent; at least 2, and a count the adapter supports for the format, which WARP does at 2, 4, 8 and 16 for each of the formats above) of the 4 by 4 texels (`size`), every sample of every texel `texel`, or sample `k`'s `per_sample[k]` where `per_sample` (one texel for each sample) is given. A multisampled texture cannot be uploaded to, so it is drawn into once, by a full-screen triangle whose fragment shader runs for each sample and writes that sample's texel; `textureLoad(t, coord, sample)` and `textureNumSamples(t)` then read it. The harness offers no sample positions: wgpu has no way to ask for them, and WGSL no built-in that states them.
  - `sampler`: point filtering of the nearest level with coordinates clamped to the edge, so a sample of a uniform texture is its texel; a comparison sampler compares with `compare` (`less` where absent; `less_equal`, `greater`, `greater_equal`, `equal`, `not_equal`, `always`, `never`).
  - `array`: a binding array's elements in order, each one of the above as the array's element type takes it (`"array": [{ "buffer": { "words": [1] } }, { "buffer": { "words": [2] } }]`): as many as a sized array's size, at least one for an unsized array.

## Replies

```json
{ "id": "…", "invocations": [ { "returned": true, "results": [[1073741824, 1082130432, 1086324736, 1090519040], [21]] }, { "returned": false, "results": [] } ],
  "resources": [ { "group": 3, "binding": 0, "words": [129, 103] } ], "adapter": "Microsoft Basic Render Driver (Dx12, …)" }
{ "id": "…", "error": "naga's WGSL front end: …" }
```

- `invocations`: per invocation, in the job's order: whether it returned (a discarded pixel did not) and its results, each member of the entry point's result structure in order (or its one result), as arguments' words are given.
- `resources`: each writable resource after the run: a storage buffer's words, a storage texture's texels of every layer row by row (a word per lane of its format); an element of a binding array with its `element`.
- `error`: why the job did not run: naga's front end or validator on the program (with the adapter's capabilities: WARP through wgpu has no `f64`), wgpu's validation, a shader compiler's error, a resource the job lacks, the timeout.

## Probes

`probes/` holds measurements of WARP itself, run through the harness (`python probes/<name>.py <wgsl-run.exe> <dxcompiler.dll>`):

- `sin.py`: WARP's `sin` of 64 arguments from 269 to 55424 against the correctly rounded `f32` sine, and how far apart that leaves `fract(sin(x) * 43758.5453)`, the hash iridia-dxbc's `exec_wgsl` test finds Unity's value noise built on.
- `div.py`: WARP's `a / b` against the correctly rounded `f32` quotient (all 4,096 equal), and how far apart that leaves `fract(saturate(a / b) * 65025)`, the depth pack Elden Ring's ambient occlusion uses.
- `msaa.py`: multisampled textures of each class (float, unsigned, signed, depth) at 1, 2, 4, 8 and 16 samples, each sample given its own texel and read back with `textureLoad` and `textureNumSamples` (2 to 16 work; 1 is refused).
- `mad.py`: which of three `f32` evaluations of `1.0 - (a * 0.85 + 0.15)` WARP's answer through WGSL equals: the unfused one, `fma` then subtract, or DXC's folded `(1 - 0.15) - a * 0.85` (all 4,096 equal that one).
