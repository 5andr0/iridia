//! A lifted program on both executors: `iridia-exec-ir`'s run, and `iridia-exec`'s ([`ProgramShader::from_ir`]) on the
//! same inputs in the D3D register view, which must agree with it: every output lane it writes, the clip position, the
//! discard, and every sample it asks for. The second run's samples are answered with the first run's answers, so a
//! test's sampler is asked once per sample.

#![allow(dead_code, reason = "each test file uses part of it")]

use iridia_exec::{PixelShader, ProgramShader};
use iridia_exec_ir::{Bound, Lanes, Lod, Run, Sample};
use iridia_ir::{
    Program,
    interface::{ResourceId, ResourceKind, Stage, SystemValue},
};

/// The lift of a token stream through the passes every consumer of a lift takes, validated and written as WGSL.
pub fn lifted(bytes: &[u8]) -> Program {
    let mut p = iridia_d3d9_ir::lift(bytes).unwrap();
    iridia_passes::prepare(&mut p).unwrap();
    iridia_ir::validate(&p).unwrap_or_else(|e| panic!("{e:?}\n{}", iridia_ir::text::print(&p.entry)));
    iridia_naga::wgsl(&p, &iridia_naga::Profile::BROWSER).unwrap();
    p
}

pub fn no_sample(_: Sample) -> Result<[f32; 4], String> {
    Err("unexpected sample".into())
}

/// The run of `p` on these input slots and constant rows, every sample answered by `sampler`, once both executors
/// agree on it.
pub fn execute(
    p: &Program,
    inputs: &[Lanes],
    constants: &[Lanes],
    sampler: &mut dyn FnMut(Sample) -> Result<[f32; 4], String>,
) -> Run {
    let bank = p
        .interface
        .resources
        .iter()
        .find(|r| matches!(r.kind, ResourceKind::ConstantBuffer { .. }));
    let buffers: Vec<(ResourceId, &[Lanes])> = bank.map(|r| (r.id, constants)).into_iter().collect();
    let mut asked = Vec::new();
    let run = iridia_exec_ir::run(
        p,
        &mut Bound {
            inputs,
            buffers: &buffers,
            state: &[],
            sampler: &mut |s: Sample| {
                let answer = sampler(s);
                asked.push((s, answer.clone()));
                answer
            },
            denormals: iridia_ir::Denormals::Any,
        },
    )
    .unwrap();
    agree(p, &run, inputs, constants, &asked);
    run
}

/// Each input slot's lanes as `registers` (by D3D register) hold them.
pub fn slots(p: &Program, registers: &[Lanes]) -> Vec<Lanes> {
    p.interface
        .inputs
        .iter()
        .map(|v| {
            let d = v.provenance.d3d.expect("an input with a D3D register");
            let mut lanes = [0; 4];
            for (k, lane) in (0..4).filter(|l| d.mask >> l & 1 != 0).enumerate() {
                lanes[k] = registers[d.register as usize][lane];
            }
            lanes
        })
        .collect()
}

/// The clip position `run` gives, as f32 bits.
pub fn clip(p: &Program, run: &Run) -> Lanes {
    let slot = p
        .interface
        .outputs
        .iter()
        .position(|v| v.system == Some(SystemValue::ClipPosition))
        .expect("a clip position");
    run.outputs[slot].expect("the clip position is written")
}

/// Each output's lanes of `run` where its D3D register and lanes put them, as `iridia-exec` gives its outputs.
pub fn by_register(p: &Program, run: &Run) -> Vec<Lanes> {
    let mut registers = Vec::new();
    for (v, lanes) in p.interface.outputs.iter().zip(&run.outputs) {
        let d = v.provenance.d3d.expect("an output with a D3D register");
        let register = d.register as usize;
        if registers.len() <= register {
            registers.resize(register + 1, [0; 4]);
        }
        let lanes = lanes.unwrap_or_default();
        for (k, lane) in (0..4).filter(|l| d.mask >> l & 1 != 0).enumerate() {
            registers[register][lane] = lanes[k];
        }
    }
    registers
}

/// The D3D register of resource `id`.
fn register(p: &Program, id: ResourceId) -> u32 {
    p.interface
        .resources
        .iter()
        .find(|r| r.id == id)
        .and_then(|r| r.provenance.d3d)
        .unwrap_or_else(|| panic!("resource {} has no D3D register", id.0))
        .register
}

fn same_lod(exec: iridia_exec::Lod, ir: Lod) -> bool {
    match (exec, ir) {
        (iridia_exec::Lod::Implicit, Lod::Auto) => true,
        (iridia_exec::Lod::Bias(a), Lod::Bias(b)) | (iridia_exec::Lod::Level(a), Lod::Exact(b)) => a == b,
        _ => false,
    }
}

/// That `iridia-exec`'s execution of `p` agrees with `run`, `iridia-exec-ir`'s, on the same inputs and constants, its
/// samples answered as `asked` records.
fn agree(
    p: &Program,
    run: &Run,
    inputs: &[Lanes],
    constants: &[Lanes],
    asked: &[(Sample, Result<[f32; 4], String>)],
) {
    let text = iridia_ir::text::print(&p.entry);
    let shader = ProgramShader::from_ir(p.clone()).unwrap_or_else(|e| panic!("iridia-exec: {e}\n{text}"));
    // each input slot's lanes at its register's lanes
    let mut registers = vec![[0u32; 4]; shader.input_count()];
    for (v, lanes) in p.interface.inputs.iter().zip(inputs) {
        let d = v.provenance.d3d.expect("an input with a D3D register");
        for (k, lane) in (0..4).filter(|l| d.mask >> l & 1 != 0).enumerate() {
            registers[d.register as usize][lane] = lanes[k];
        }
    }
    let buffers: Vec<(u32, &[Lanes])> = shader
        .constant_buffers()
        .iter()
        .map(|&(slot, _)| (slot, constants))
        .collect();
    let mut replay = asked.iter();
    let mut sampler = |s: iridia_exec::Sample| -> Result<[f32; 4], String> {
        let (wanted, answer) = replay
            .next()
            .unwrap_or_else(|| panic!("iridia-exec samples more than iridia-exec-ir\n{text}"));
        assert_eq!(
            (s.texture, s.sampler, s.uv, wanted.lanes),
            (
                register(p, wanted.texture),
                register(p, wanted.sampler),
                [wanted.coord[0], wanted.coord[1]],
                2
            ),
            "the sample iridia-exec asks for\n{text}"
        );
        assert!(
            same_lod(s.lod, wanted.lod),
            "level of detail {:?} against {:?}\n{text}",
            s.lod,
            wanted.lod
        );
        answer.clone()
    };
    let (execution, clip) = if p.stage == Stage::Vertex {
        let (e, clip) = shader
            .execute_vertex(&registers, &buffers, &mut sampler)
            .unwrap_or_else(|e| panic!("iridia-exec: {e}\n{text}"));
        (e, Some(clip))
    } else {
        let e = shader
            .execute(&registers, &buffers, &mut sampler)
            .unwrap_or_else(|e| panic!("iridia-exec: {e}\n{text}"));
        (e, None)
    };
    assert_eq!(
        replay.len(),
        0,
        "iridia-exec samples less than iridia-exec-ir\n{text}"
    );
    assert_eq!(execution.discarded, run.killed, "discarded\n{text}");
    for (slot, v) in p.interface.outputs.iter().enumerate() {
        let d = v.provenance.d3d.expect("an output with a D3D register");
        let register = d.register as usize;
        let reference = run.outputs[slot].unwrap_or_default();
        let written = execution.output_masks.get(register).copied().unwrap_or(0);
        for (k, lane) in (0..4).filter(|l| d.mask >> l & 1 != 0).enumerate() {
            if written >> lane & 1 != 0 {
                assert_eq!(
                    execution.outputs[register][lane], reference[k],
                    "output {}{} lane {lane}\n{text}",
                    v.label.name, v.label.index
                );
            }
        }
        if v.system == Some(SystemValue::ClipPosition) {
            assert_eq!(clip, Some(reference), "the clip position\n{text}");
        }
    }
}
