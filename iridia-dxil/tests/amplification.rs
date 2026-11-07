#![allow(clippy::unwrap_used)]
//! The amplification fixtures lifted onto the IR ([`iridia_dxil::ir`]) and run: an amplification program's launches (the
//! first invocation's thread groups, the payload every invocation wrote before `DispatchMesh`'s barrier), the meshlets
//! of the mesh program each launch runs, and a payload laid out in bytes as Direct3D lays it out (`payload_layout`,
//! `payload_halves`), after vkd3d's `mesh.shader_test`.
use iridia_exec::amplification::{AmplificationShader, Launch};
use iridia_exec::mesh::MeshShader;
use iridia_ir::{
    Program,
    interface::{AmplificationFacts, Stage, StageFacts},
};

fn program(name: &str) -> Program {
    let bytes = std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name),
    )
    .unwrap();
    let mut p = iridia_dxil::ir::lift(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
    iridia_passes::structurize(&mut p.entry).unwrap();
    iridia_passes::break_flags(&mut p.entry).unwrap();
    iridia_ir::validate(&p).unwrap_or_else(|e| panic!("{name}: {e:?}"));
    p
}

fn nothing(_: iridia_exec::Sample) -> Result<[f32; 4], String> {
    Err("no sampling in these programs".into())
}

fn no_constants() -> std::collections::BTreeMap<(u32, u32), Vec<[u32; 4]>> {
    std::collections::BTreeMap::new()
}

/// The first invocation's `DispatchMesh` counts (two thread groups, not one), and the barrier it is collects the lane
/// each of the four invocations wrote; each mesh group reads that payload and draws the triangle its group id picks.
#[test]
fn an_amplification_program_launches_its_first_invocations_groups_with_every_invocations_payload() {
    let a = program("amplification.dxil");
    assert_eq!(a.stage, Stage::Amplification);
    // the target dxc was given (`-T as_6_5`), which the program header names by its shader kind, 14
    assert_eq!(a.provenance.target, "as_6_5");
    assert_eq!(program("amplification_mesh.dxil").provenance.target, "ms_6_5");
    assert_eq!(a.workgroup, Some([4, 1, 1]));
    assert_eq!(
        a.interface.facts,
        StageFacts::Amplification(AmplificationFacts { payload: 16 })
    );
    let shader = AmplificationShader::from_ir(&a).unwrap();
    let launches = shader
        .launches(&no_constants(), &mut nothing, [1, 1, 1], 32)
        .unwrap();
    let colour = [0.25f32, 0.5, 0.75, 1.0].map(f32::to_bits);
    assert_eq!(
        launches,
        [Launch {
            group: [0, 0, 0],
            groups: [2, 1, 1],
            payload: colour.to_vec(),
        }]
    );
    let mesh = MeshShader::from_ir(&program("amplification_mesh.dxil")).unwrap();
    assert_eq!(mesh.facts().payload, 16);
    let meshlets = shader
        .draw(&mesh, &no_constants(), &mut nothing, [1, 1, 1], 32)
        .unwrap();
    assert_eq!(meshlets.len(), 2);
    for (g, m) in (0u32..).zip(&meshlets) {
        assert_eq!(m.group, [g, 0, 0]);
        assert_eq!(m.primitives, [[[0, 1, 3], [1, 2, 3]][g as usize].to_vec()]);
        assert!(m.vertices.iter().all(|v| v[1] == colour), "{m:?}");
    }
    // a mesh program that reads a payload, drawn without one
    assert!(
        mesh.draw(&no_constants(), &mut nothing, [1, 1, 1], 32)
            .unwrap_err()
            .to_string()
            .contains("run without one")
    );
}

/// A local structure of a halfword, a doubleword, a halfword, a bool and a half is the payload's 32 bytes at offsets
/// 0, 8, 16, 20 and 24; the mesh program reads them back as its 16 halfwords.
#[test]
fn a_payload_is_laid_out_as_direct3d_lays_out_its_structure() {
    let a = program("payload_layout.dxil");
    assert_eq!(
        a.interface.facts,
        StageFacts::Amplification(AmplificationFacts { payload: 32 })
    );
    let shader = AmplificationShader::from_ir(&a).unwrap();
    let launches = shader
        .launches(&no_constants(), &mut nothing, [1, 1, 1], 32)
        .unwrap();
    let words = [0xabab, 0, 0x5566_7788, 0x1122_3344, 0xcdcd, 1, 0x3e00, 0];
    assert_eq!(launches.len(), 1);
    assert_eq!(launches[0].payload, words);
    let mesh = MeshShader::from_ir(&program("payload_layout_mesh.dxil")).unwrap();
    let meshlets = shader
        .draw(&mesh, &no_constants(), &mut nothing, [1, 1, 1], 32)
        .unwrap();
    let halves: Vec<u32> = words.iter().flat_map(|w| [w & 0xffff, w >> 16]).collect();
    for v in &meshlets[0].vertices {
        let read: Vec<u32> = v[1..].iter().flatten().copied().collect();
        assert_eq!(read, halves);
    }
}

/// Two invocations write the two halfwords of one word of a groupshared payload at once, and neither write is lost.
#[test]
fn two_halfwords_of_one_payload_word_written_at_once_are_both_kept() {
    let shader = AmplificationShader::from_ir(&program("payload_halves.dxil")).unwrap();
    let launches = shader
        .launches(&no_constants(), &mut nothing, [1, 1, 1], 32)
        .unwrap();
    assert_eq!(launches[0].payload, [0xbbbb_aaaa, 7]);
}

/// The mesh launch in the IR's text form, read back as it was printed; one in another stage, or of a mesh program's
/// own payload, is refused by the validator.
#[test]
fn a_mesh_launch_round_trips_and_belongs_to_amplification_programs() {
    let p = program("amplification.dxil");
    let text = iridia_ir::text::print(&p.entry);
    assert!(text.contains("dispatch mesh %"), "{text}");
    assert_eq!(
        iridia_ir::text::print(&iridia_ir::text::parse(&text).unwrap()),
        text
    );
    let mesh = program("amplification_mesh.dxil");
    let text = iridia_ir::text::print(&mesh.entry);
    assert!(
        text.contains(" payload\n") || text.contains(" payload\r\n"),
        "{text}"
    );
    assert_eq!(
        iridia_ir::text::print(&iridia_ir::text::parse(&text).unwrap()),
        text
    );
    let mut q = p.clone();
    q.stage = Stage::Compute;
    q.interface.stage = Stage::Compute;
    q.interface.facts = StageFacts::None;
    let problems = iridia_ir::validate(&q).unwrap_err();
    assert!(
        problems.iter().any(|e| e
            .to_string()
            .contains("a mesh launch outside an amplification program")),
        "{problems:?}"
    );
}

/// What the validator refuses of a task payload and a mesh launch: a store into a mesh program's payload, a payload
/// other than the words the facts' bytes fill, and a launch of thread groups that are not a `vec3<u32>`.
#[test]
fn a_payload_is_read_alone_and_as_large_as_the_facts_state() {
    let problems = |p: &Program| -> Vec<String> {
        iridia_ir::validate(p)
            .err()
            .unwrap_or_default()
            .iter()
            .map(ToString::to_string)
            .collect()
    };
    let refused = |p: &Program, what: &str| {
        let found = problems(p);
        assert!(found.iter().any(|e| e.contains(what)), "{what}: {found:?}");
    };
    // a store into the mesh program's payload, after its counts
    let mesh = program("amplification_mesh.dxil");
    let text = iridia_ir::text::print(&mesh.entry);
    let payload = mesh
        .entry
        .vars
        .iter()
        .find(|v| v.space == iridia_ir::Space::Payload)
        .unwrap();
    let counts = text.lines().find(|l| l.contains("mesh counts %")).unwrap();
    let operands = counts.split("mesh counts ").nth(1).unwrap();
    let (a, b) = operands.split_once(", ").unwrap();
    let indent: String = counts.chars().take_while(|c| c.is_whitespace()).collect();
    let stored = text.replacen(
        counts,
        &format!("{counts}\n{indent}{}[{a}] = {b} oob.zero", payload.name),
        1,
    );
    let mut q = mesh.clone();
    q.entry = iridia_ir::text::parse(&stored).unwrap();
    refused(&q, "a store to task payload");
    // the facts state another size
    let mut q = mesh.clone();
    let StageFacts::Mesh(facts) = &mut q.interface.facts else {
        unreachable!()
    };
    facts.payload = 32;
    refused(&q, "is not the 8 u32 words of the 32-byte payload");
    let mut q = program("amplification.dxil");
    q.interface.facts = StageFacts::Amplification(AmplificationFacts { payload: 8 });
    refused(&q, "not the 2 u32 words of the 8-byte payload");
    // thread groups of two lanes
    let a = program("amplification.dxil");
    let text = iridia_ir::text::print(&a.entry);
    let launch = text.lines().find(|l| l.contains("dispatch mesh %")).unwrap();
    let groups = launch
        .split("dispatch mesh ")
        .nth(1)
        .unwrap()
        .split(", ")
        .next()
        .unwrap();
    let indent: String = launch.chars().take_while(|c| c.is_whitespace()).collect();
    let id: u32 = text
        .lines()
        .filter_map(|l| l.trim().strip_prefix('%')?.split(':').next()?.parse::<u32>().ok())
        .max()
        .unwrap()
        + 1;
    let narrowed = text.replacen(
        launch,
        &format!(
            "{indent}%{id}: vec2<u32> = {groups}.xy\n{}",
            launch.replacen(groups, &format!("%{id}"), 1)
        ),
        1,
    );
    let mut q = a.clone();
    q.entry = iridia_ir::text::parse(&narrowed).unwrap_or_else(|e| panic!("{e}\n{narrowed}"));
    refused(&q, "a mesh launch of other thread groups than a vec3<u32>");
}

/// `promote` drops a private variable declared ahead of the payload's and renumbers the variables that stay, the mesh
/// launch's payload among them: the launches are the program's own.
#[test]
fn promote_renumbers_the_variable_a_mesh_launch_hands_over() {
    let a = program("amplification.dxil");
    let text = iridia_ir::text::print(&a.entry);
    let header = text.lines().find(|l| l.starts_with("fn ")).unwrap();
    let spared = text.replacen(header, &format!("{header}\n  var spare: u32"), 1);
    let mut q = a.clone();
    q.entry = iridia_ir::text::parse(&spared).unwrap();
    assert_eq!(q.entry.vars.len(), a.entry.vars.len() + 1);
    iridia_passes::promote(&mut q.entry).unwrap();
    let mut a = a;
    iridia_passes::promote(&mut a.entry).unwrap();
    assert_eq!(q.entry.vars, a.entry.vars);
    iridia_ir::validate(&q).unwrap_or_else(|e| panic!("{e:?}"));
    let launches = |p: &Program| {
        AmplificationShader::from_ir(p)
            .unwrap()
            .launches(&no_constants(), &mut nothing, [1, 1, 1], 32)
            .unwrap()
    };
    assert_eq!(launches(&q), launches(&a));
}
