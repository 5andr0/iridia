//! Material specialisation against the old specialiser and both unmodified executions.
#![allow(clippy::unwrap_used)]

use iridia_exec::PixelShader;
use iridia_ir::{
    Program,
    interface::{ResourceId, ResourceKind},
};
use iridia_passes::{SpecialiseError, Words, feature_words, specialise};

fn cases(name: &str) -> Vec<(iridia::Program, Program)> {
    let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../iridia/tests/fixtures");
    ["dxbc", "dxil"]
        .into_iter()
        .map(|ext| {
            let bytes = std::fs::read(base.join(format!("specialise_{name}.{ext}"))).unwrap();
            if ext == "dxbc" {
                (
                    iridia_dxbc::lift(&bytes).unwrap(),
                    iridia_dxbc::ir::lift(&bytes).unwrap(),
                )
            } else {
                (
                    iridia_dxil::lift(&bytes).unwrap(),
                    iridia_dxil::ir::lift(&bytes).unwrap(),
                )
            }
        })
        .collect()
}

fn buffer(p: &Program) -> ResourceId {
    p.interface
        .resources
        .iter()
        .find(|r| matches!(r.kind, ResourceKind::ConstantBuffer { .. }))
        .unwrap()
        .id
}

fn run(p: &Program, inputs: &[[u32; 4]], id: ResourceId, row: [u32; 4]) -> iridia_exec_ir::Run {
    let rows = [row];
    let buffers = [(id, rows.as_slice())];
    let mut sampler = |_| Ok([0.0; 4]);
    iridia_exec_ir::run(
        p,
        &mut iridia_exec_ir::Bound {
            inputs,
            buffers: &buffers,
            state: &[],
            sampler: &mut sampler,
            denormals: iridia_ir::Denormals::Preserve,
        },
    )
    .unwrap()
}

fn check(name: &str, offset: u32, values: &[u32]) {
    for (old, ir) in cases(name) {
        let original = ir.clone();
        let id = buffer(&ir);
        let old_id = old
            .interface
            .resources
            .iter()
            .find(|r| matches!(r.kind, ResourceKind::ConstantBuffer { .. }))
            .unwrap()
            .id;
        for &value in values {
            // An ordinary scale is supplied too: it must stay a uniform, including when it shares a vector load
            // with the feature word. Vary it at execution time after specialising.
            let words = Words::from([((id, offset), value), ((id, 4), 0.5f32.to_bits())]);
            let old_words =
                iridia::specialise::Words::from([((old_id, offset), value), ((old_id, 4), 0.5f32.to_bits())]);
            let result = specialise(&ir, &words)
                .unwrap_or_else(|e| panic!("{name}: {e}\n{}", iridia_ir::text::print(&ir.entry)));
            let before = iridia_ir::text::print(&ir.entry);
            let old_result = iridia::specialise::specialise(&old, &old_words).unwrap();
            assert_eq!(result.words, Words::from([((id, offset), value)]));
            assert_eq!(
                old_result.words,
                iridia::specialise::Words::from([((old_id, offset), value)])
            );
            assert!(result.folded > 0);
            let old_source = iridia_exec::ProgramShader::new(old.clone()).unwrap();
            let old_specialised = iridia_exec::ProgramShader::new(old_result.program).unwrap();
            for i in 0..64u32 {
                let inputs = [
                    [0; 4],
                    [
                        (i as f32 / 64.).to_bits(),
                        0.25f32.to_bits(),
                        0.5f32.to_bits(),
                        ((63 - i) as f32 / 64.).to_bits(),
                    ],
                ];
                let mut row = [0, (i as f32 / 32.).to_bits(), 0, 0];
                row[usize::try_from(offset / 4).unwrap()] = value;
                let a = run(&ir, &inputs, id, row);
                let b = run(&result.program, &inputs, id, row);
                assert_eq!(
                    a,
                    b,
                    "{name}, value={value}, input={i}\n{}",
                    iridia_ir::text::print(&result.program.entry)
                );
                for shader in [&old_source, &old_specialised] {
                    let mut reads =
                        iridia_exec::probe::AnsweredReads::new(|_: &iridia_exec::ImageSample<u32>| {
                            Ok([0.; 4])
                        });
                    let old_run = shader.execute(&inputs, &[(0, &[row])], &mut reads).unwrap();
                    assert_eq!(
                        old_run.outputs,
                        a.outputs.iter().map(|v| v.unwrap_or([0; 4])).collect::<Vec<_>>()
                    );
                }
            }
            assert_eq!(ir, original);
            assert_eq!(iridia_ir::text::print(&ir.entry), before);
        }
    }
}

#[test]
fn supplied_boolean_word_takes_a_select_without_freezing_its_neighbour() {
    check("word", 0, &[0, 1]);
}

#[test]
fn signed_feature_comparisons_take_both_paths() {
    check("integer", 8, &[(-1i32).cast_unsigned(), 5]);
}

#[test]
fn unsupported_feature_arithmetic_returns_a_typed_refusal() {
    for (_, ir) in cases("root") {
        let words = Words::from([((buffer(&ir), 4), 4f32.to_bits())]);
        assert_eq!(feature_words(&ir, &words).unwrap(), words);
        assert!(matches!(
            specialise(&ir, &words),
            Err(SpecialiseError::UnresolvedDecision { .. })
        ));
        assert!(specialise(&ir, &Words::new()).is_ok());
    }
}

#[test]
fn malformed_words_have_distinct_typed_errors_and_leave_the_source_unchanged() {
    let (_, ir) = cases("word").remove(0);
    let saved = ir.clone();
    let id = buffer(&ir);
    assert!(matches!(
        specialise(&ir, &Words::from([((ResourceId(u32::MAX), 0), 0)])),
        Err(SpecialiseError::MissingResource(_))
    ));
    assert!(matches!(
        specialise(&ir, &Words::from([((id, 1), 0)])),
        Err(SpecialiseError::WordOutOfBounds { .. })
    ));
    assert!(matches!(
        specialise(&ir, &Words::from([((id, u32::MAX - 3), 0)])),
        Err(SpecialiseError::WordOutOfBounds { .. })
    ));
    assert_eq!(ir, saved);
}

fn float_decision() -> Program {
    let (_, mut ir) = cases("word").remove(0);
    let id = buffer(&ir).0;
    ir.entry = iridia_ir::text::parse(&format!(
        "iridia-ir 1
fn main {{
  %0: f32 = buffer r{id} + 0 oob.zero
  %1: f32 = f32(0.0)
  %2: bool = fcmp.ne %0, %1
  %3: f32 = input 1.x
  %4: f32 = f32(2.0)
  %5: f32 = select %2, %3, %4
  output 0.x = %5
  return
}}
"
    ))
    .unwrap();
    iridia_ir::validate(&ir).unwrap();
    ir
}

#[test]
fn float_feature_comparisons_respect_nan_signed_zero_and_denormal_controls() {
    let mut ir = float_decision();
    let id = buffer(&ir);
    let inputs = [[0; 4], [3f32.to_bits(); 4]];
    for policy in [iridia_ir::Denormals::Flush, iridia_ir::Denormals::Preserve] {
        ir.float.f32 = policy;
        for word in [0, 0x8000_0000, 1, 0x8000_0001, 1f32.to_bits(), f32::NAN.to_bits()] {
            let result = specialise(&ir, &Words::from([((id, 0), word)])).unwrap();
            assert!(result.unread.contains(&id));
            assert_eq!(
                run(&ir, &inputs, id, [word, 0, 0, 0]),
                run(&result.program, &inputs, id, [word, 0, 0, 0])
            );
        }
    }
    ir.float.f32 = iridia_ir::Denormals::Any;
    assert!(matches!(
        specialise(&ir, &Words::from([((id, 0), 1)])),
        Err(SpecialiseError::UnresolvedDecision { .. })
    ));
}

proptest::proptest! {
    #[test]
    fn arbitrary_feature_bits_preserve_runtime_inputs(flag: u32, scale in -64i16..64, x: u8, w: u8) {
        let (_, ir) = cases("word").remove(0);
        let id = buffer(&ir);
        let result = specialise(&ir, &Words::from([((id,0),flag), ((id,4),1f32.to_bits())])).unwrap();
        proptest::prop_assert_eq!(&result.words, &Words::from([((id,0),flag)]));
        let inputs = [[0;4], [(f32::from(x)/256.).to_bits(), 0.25f32.to_bits(), 0.5f32.to_bits(), (f32::from(w)/256.).to_bits()]];
        let row = [flag, (f32::from(scale)/8.).to_bits(), 0, 0];
        proptest::prop_assert_eq!(run(&ir, &inputs, id, row), run(&result.program, &inputs, id, row));
    }
}
