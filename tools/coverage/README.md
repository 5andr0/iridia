# iridia-coverage

What iridia and dx read, list, lift, run and write, per construct (an opcode, or a stage of a bytecode), measured by running them over two test suites: [vkd3d](https://gitlab.winehq.org/wine/vkd3d)'s HLSL shader tests and LLVM's [offload-test-suite](https://github.com/llvm/offload-test-suite). Nothing in the table is written by hand: every number comes from the case files in [`suites`](suites) and the runs.

Part of [iridia](https://github.com/Kalmarv/iridia). Not published: it is the workspace's own measuring tool.

## The table

`iridia-coverage table [<out dir>] [<offload-golden-images checkout>]` runs both suites and writes `coverage.json` (the data the demo site reads) and `coverage.md` (a summary) to `docs/coverage` or the directory given. CI makes it again on every change and fails when it differs from the one committed, so the table in the repository is the one the code makes.

The table grades iridia's IR path (`graded` in the JSON): each bytecode's IR lift (`iridia-dxbc`'s, `iridia-dxil`'s, `iridia-d3d9-ir`), `iridia_passes::prepare` (`structurize` where the lift makes a graph, `promote`, `simplify`, `break_flags`) and the IR validated, run in `iridia-exec` on the IR. Each case file runs again through the lift the IR replaces (DXBC and DXIL lifted into naga modules; a Direct3D 9 program's WGSL read back by naga, its IR drawn again and held to it), each row's `reference_status`, until that lift is removed.

The table is made of parts, each case file through one path, or one shard of it (`--shard <index>/<count>`: the cases whose number leaves `index` modulo `count`, so one file's runs fall to different shards): `iridia-coverage part vkd3d|offload <cases.json.gz> <out.json> [<golden images>] [--reference] [--shard <index>/<count>]` runs one, and `table --parts <dir>` makes the table of the parts in a folder instead of running them, a file's shards put together in the order of its cases. CI builds the tool once, runs the parts side by side (the matrix in `.github/workflows/ci.yml` says how many shards each file has) and makes the table of them; `table` alone runs them one after another.

For each construct, over the suites' programs that use it:

- **decoded**: dx reads the program (`d3d9-reader`, `dxbc-reader`, `dxil-reader`);
- **listed**: dx's listing is Microsoft's, line for line (`D3DDisassemble`, `dxc -dumpbin`, from the `.listings.json.gz` files), or unchecked where there is none;
- **lifted**: iridia lifts it onto its IR and through the passes;
- **wgsl**: the WGSL `iridia-naga` writes of the IR (for browsers) is read back and validated by naga: information, which decides no run;
- **executed**: some run or test using it passes in `iridia-exec` on the IR (`pass`), or every one using it that got as far as running failed (`fail`);
- **proved by**: the runs and tests that passed.

For each vkd3d run (a test file at a shader model on one of vkd3d's Windows runners, d3d9, d3d11, d3d12 and d3d12-dxc) and each offload test, the status, its group and why. The JSON names them too (`statuses`, `groups`), and each suite's `reasons` counts what is left by status and class of reason, with an example.

| Status | Group | Means |
|---|---|---|
| `pass` | pass | every probe holds (vkd3d), or the test's RUN lines all pass as lit runs them (offload). A vkd3d run whose programs leave it open whether 32-bit denormals are flushed (Direct3D 9's) and that does not hold as run is run again with them flushed; it passes when that holds and breaks no probe that held |
| `fail` | gap | a probe reads back something other than the test expects, or the suite's own check of the output fails |
| `error` | gap | iridia took the programs and could not run them: a pass, the IR's validation or the executor failed, so nothing after it is judged |
| `refused` | gap | iridia refuses the program by name (a construct it does not lift or run) |
| `not-modelled` | runner-limit | the test needs pipeline state or an offloader step the runner does not model, or a feature a Direct3D 12 device can have and iridia's does not (offload: acceleration structures, ray tracing pipelines, wave sizes but 32, barycentrics, stencil reference, shading rate, several viewports), named in the detail |
| `not-applicable` | not-applicable | offload: lit runs the test on no Direct3D 12 device in its DirectX configuration with DXC: a `REQUIRES` no device's features meet (`Vulkan`, `Clang`), or an `UNSUPPORTED` the configuration meets (`DirectX`, `DXC`, `!Vulkan`) |
| `compiler-refused` | not-applicable | Microsoft's compiler made no program of the shader |
| `winbug` | not-applicable | vkd3d: every probe deciding the run is one vkd3d's test marks `winbug` where Windows itself reads otherwise |
| `xfail` | expected | offload: a line fails where the test's `XFAIL` expects a failure |
| `xpass` | unexpected-pass | offload: every line passes where the test's `XFAIL` expects a failure (lit counts it as a failure) |

A run's status is its most severe probe's: `fail`, then `error`, `refused`, `not-modelled`, `compiler-refused`, among the probes of the sections vkd3d's runner runs at that model (`selected`). The case files also hold the runs at the models in a test file's range that vkd3d's own runs leave out, and the sections it leaves out of the runs it makes: these are extended, the same expectations at a model where they need not hold on any device (fxc places registers and folds constants differently at shader model 5.1, say). A run vkd3d runs none of is `selected: false`, its status that of all its probes; the probes left out of the others give their `extended_status`. `totals` and `probe_totals` count the selected runs and probes by status, `extended_totals` and `extended_probe_totals` the others, so a run that fails on one probe of forty shows as one failing probe and thirty-nine others.

A Direct3D 9 draw follows Direct3D 9's rules where a later API's differ: it samples pixel (x, y) at (x, y), the viewport half a pixel on as wined3d places it (the IR's `vPos` is the window position half a pixel back), and its flat shade mode makes the colour inputs the provoking vertex's. Its programs run on the IR in a register view where each varying has a register of its own (the colour and texture coordinate inputs of shader models 1 and 2 share register numbers, and `vPos` and `vFace` have none). On the reference path they run as their WGSL read back by naga, and the draw runs again from the IR: where that writes otherwise than the WGSL's draw, the run is an `error`, the detail saying where.

A probe after a step that failed is judged only when it reads nothing that step would have written: a resource a failed draw or dispatch would have written is unknown, and so is anything a later draw or dispatch makes reading it.

## The runners

`iridia-coverage vkd3d <cases.json.gz> [--reference]` and `iridia-coverage offload <cases.json.gz> [<golden images>] [--reference]` run one suite on the IR (or the reference path) and print its runs by status, then the runs that do not pass by why, each kind with an example. The device they run on (`device`) holds a test's resources and pipeline state the way Direct3D does and draws and dispatches with `iridia-exec`; `front` is the one place they learn how iridia reads, lifts and writes a program.

The case files and how to make them again are in [`suites/vkd3d`](suites/vkd3d) and [`suites/offload`](suites/offload). The offload image tests compare with [offload-golden-images](https://github.com/llvm/offload-golden-images) at the commit `suites/offload/README.md` names; without a checkout, or with a golden image that is not the one the case file names (by its SHA-256), they are `not-modelled`, the detail saying which.
