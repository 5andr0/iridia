# iridia-coverage

What iridia and dx read, list, lift, run and write, per construct (an opcode, or a stage of a bytecode), measured by running them over two test suites: [vkd3d](https://gitlab.winehq.org/wine/vkd3d)'s HLSL shader tests and LLVM's [offload-test-suite](https://github.com/llvm/offload-test-suite). Nothing in the table is written by hand: every number comes from the case files in [`suites`](suites) and the runs.

Part of [iridia](https://github.com/Kalmarv/iridia). Not published: it is the workspace's own measuring tool.

## The table

`iridia-coverage table [<out dir>] [<offload-golden-images checkout>]` runs both suites and writes `coverage.json` (the data the demo site reads) and `coverage.md` (a summary) to `docs/coverage` or the directory given. CI makes it again on every change and fails when it differs from the one committed, so the table in the repository is the one the code makes.

For each construct, over the suites' programs that use it:

- **decoded**: dx reads the program (`d3d9-reader`, `dxbc-reader`, `dxil-reader`);
- **listed**: dx's listing is Microsoft's, line for line (`D3DDisassemble`, `dxc -dumpbin`, from the `.listings.json.gz` files), or unchecked where there is none;
- **lifted**: iridia lifts it through its public pipeline;
- **wgsl**: the WGSL iridia writes of it is read back by naga;
- **executed**: some run or test using it passes in `iridia-exec` (`pass`), or every one using it that got as far as running failed (`fail`);
- **proved by**: the runs and tests that passed.

For each vkd3d run (a test file at a shader model on one of vkd3d's Windows runners, d3d9, d3d11, d3d12 and d3d12-dxc) and each offload test, the status and why:

| Status | Means |
|---|---|
| `pass` | every probe holds (vkd3d), or the test's RUN lines all pass as lit runs them (offload). A vkd3d run whose programs leave it open whether 32-bit denormals are flushed (Direct3D 9's) and that does not hold as run is run again with them flushed; it passes when that holds and breaks no probe that held |
| `fail` | a probe reads back something other than the test expects, or the suite's own check of the output fails; or a program iridia lifted was refused by a later stage (its WGSL, the executor) |
| `error` | a step the test expects to work fails in iridia (a lift, a run), so nothing after it is judged |
| `refused` | iridia refuses the program by name (a construct it does not lift or run) |
| `not-modelled` | the test needs pipeline state the runner does not model (named in the detail) |
| `compiler-refused` | Microsoft's compiler made no program of the shader |
| `unsupported`, `xfail`, `xpass` | offload: lit's own outcomes from the test's `REQUIRES`, `UNSUPPORTED` and `XFAIL` |

A run's status is its most severe probe's: `fail`, then `error`, `refused`, `not-modelled`, `compiler-refused`, among the probes of the sections vkd3d's runner runs at that model (`selected`). The case files also hold the runs at the models in a test file's range that vkd3d's own runs leave out, and the sections it leaves out of the runs it makes: these are extended, the same expectations at a model where they need not hold on any device (fxc places registers and folds constants differently at shader model 5.1, say). A run vkd3d runs none of is `selected: false`, its status that of all its probes; the probes left out of the others give their `extended_status`. `totals` and `probe_totals` count the selected runs and probes by status, `extended_totals` and `extended_probe_totals` the others, so a run that fails on one probe of forty shows as one failing probe and thirty-nine others.

A probe after a step that failed is judged only when it reads nothing that step would have written: a resource a failed draw or dispatch would have written is unknown, and so is anything a later draw or dispatch makes reading it.

## The runners

`iridia-coverage vkd3d <cases.json.gz>` and `iridia-coverage offload <cases.json.gz> [<golden images>]` run one suite and print its runs by status, then the runs that do not pass by why, each kind with an example. The device they run on (`device`) holds a test's resources and pipeline state the way Direct3D does and draws and dispatches with `iridia-exec`; `front` is the one place they learn how iridia reads, lifts and writes a program.

The case files and how to make them again are in [`suites/vkd3d`](suites/vkd3d) and [`suites/offload`](suites/offload). The offload image tests compare with [offload-golden-images](https://github.com/llvm/offload-golden-images) at the commit `suites/offload/README.md` names; without a checkout, or with a golden image that is not the one the case file names (by its SHA-256), they are `not-modelled`, the detail saying which.
