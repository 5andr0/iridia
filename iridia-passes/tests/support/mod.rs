//! What the pass tests share: a pixel program around a function, and a run of it in the IR executor.

#![allow(dead_code, reason = "each test file uses part of it")]

use iridia_ir::{
    FloatEnv, Function, Program, Provenance,
    interface::{
        Interface, Label, Resource, ResourceId, ScalarType, ShaderFormat, Stage, StageFacts, SystemValue,
        Value, Varying, VaryingProvenance,
    },
};

fn varying(name: &str, index: u32, value: (ScalarType, u8), system: Option<SystemValue>) -> Varying {
    Varying {
        label: Label {
            name: name.into(),
            index,
        },
        system,
        value: Value {
            scalar: value.0,
            width: value.1,
        },
        interp: None,
        used: true,
        provenance: VaryingProvenance::default(),
    }
}

/// A pixel program of `entry`, with `TEXCOORD` inputs and `SV_Target` outputs of the shapes given.
pub fn program(
    entry: Function,
    inputs: &[(ScalarType, u8)],
    outputs: &[(ScalarType, u8)],
    resources: Vec<Resource>,
) -> Program {
    let n = |i: usize| u32::try_from(i).unwrap_or(u32::MAX);
    Program {
        stage: Stage::Pixel,
        entry,
        interface: Interface {
            stage: Stage::Pixel,
            inputs: inputs
                .iter()
                .enumerate()
                .map(|(i, v)| varying("TEXCOORD", n(i), *v, None))
                .collect(),
            outputs: outputs
                .iter()
                .enumerate()
                .map(|(i, v)| varying("SV_Target", n(i), *v, Some(SystemValue::Target(n(i)))))
                .collect(),
            resources,
            facts: StageFacts::default(),
        },
        conventions: iridia_ir::Conventions::D3D,
        float: FloatEnv::default(),
        provenance: Provenance {
            format: ShaderFormat::D3d9,
            target: "test".into(),
            lifter: "test".into(),
            lifter_version: 1,
            source_hash: None,
            instruction_count: None,
            policies: Vec::new(),
        },
        notes: Vec::new(),
    }
}

/// A program of the text's function; the text must be in the printer's own form.
pub fn from_text(
    text: &str,
    inputs: &[(ScalarType, u8)],
    outputs: &[(ScalarType, u8)],
    resources: Vec<Resource>,
) -> Program {
    let entry = iridia_ir::text::parse(text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    assert_eq!(
        iridia_ir::text::print(&entry),
        text,
        "the fixture is in the printer's own form"
    );
    let p = program(entry, inputs, outputs, resources);
    iridia_ir::validate(&p).unwrap_or_else(|e| panic!("{e:?}\n{text}"));
    p
}

/// Each output slot's lanes (`None` for one never stored) and whether the run killed.
pub type Outcome = (Vec<Option<[u32; 4]>>, bool);

/// The outcome of a run, or the run's refusal.
pub fn run(p: &Program, inputs: &[[u32; 4]], state: &[(ResourceId, [u32; 4])]) -> Result<Outcome, String> {
    let mut sampler = |_| Err::<[f32; 4], String>("no textures".into());
    let mut bound = iridia_exec_ir::Bound {
        inputs,
        buffers: &[],
        state,
        sampler: &mut sampler,
    };
    let r = iridia_exec_ir::run(p, &mut bound)?;
    Ok((r.outputs, r.killed))
}
