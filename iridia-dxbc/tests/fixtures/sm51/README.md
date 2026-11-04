# SM5.1 execution fixtures

Each HLSL header contains its fxc 10.0.26100.0 and dxc (Windows SDK 10.0.26100.0) commands.
`.51.asm` is the corresponding `fxc -dumpbin name.51.dxbc` output, with trailing whitespace removed. `spaces` also builds as
SM5.0 by omitting register spaces; the other sources use dynamic descriptor arrays that SM5.0
does not express. The HLSL is our own test data, with no game programs.

`sm51_oracle.rs` checks every executable instruction's opcode and operand registers/indices,
including relative-index registers and components, and the non-uniform marker, against the fxc
dump. `sm51_exec.rs` lifts and validates every build, runs the same inputs, and requires exact
32-bit equality of outputs. It also checks the saved expected outputs. Regenerate those only
after inspecting the cross-build comparison (`SM51_WRITE_EXPECTED=1`).

Each `.inputs.json` describes six independent runs. Constants are rows of four raw u32 words;
an array has another outer descriptor dimension. Buffers are raw words, likewise nested for
descriptor arrays. Images are arrays of descriptors, each a row-major 2x2 image of four-word
texels with the format in `image_formats`. Samplers use point filtering and the named address
mode; the comparison sampler uses LESS_EQUAL and border colour zero. Writable buffers start
with the saved words for every run. Pixel arguments are register rows; compute runs specify
dispatch groups. Constant-buffer registers/spaces and other binding locations come from each
compiled shader's reflection, not from a shared binding slot convention.

`.expected.json` has one output per run: pixel `values`, or compute `buffers` keyed by
`resource_name:descriptor_index` and image writes keyed by `resource_name:index:[x,y,z]:layer`.
Buffer snapshots include untouched sentinel words, so writing the wrong descriptor changes the
comparison. Inputs and output words can also be fed to a D3D12 harness without the CPU lifter.
