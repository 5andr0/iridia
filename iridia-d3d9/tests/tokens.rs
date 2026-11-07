#![allow(clippy::unwrap_used)]
//! Hand-worked SM3 tokens: what the naga lift declares, keeps and refuses. The programs' answers, worked from the SDK
//! instruction definitions, are held on the IR, on both executors (iridia-d3d9-ir's tests/tokens.rs).
use iridia::{ResourceKind, ShaderFormat};
use iridia_d3d9::{lift, read};

fn program(body: &[u32]) -> Vec<u8> {
    [&[0xffff_0300][..], body, &[0xffff][..]]
        .concat()
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect()
}
fn instruction(op: u32, args: &[u32]) -> Vec<u32> {
    let mut words = vec![op | (args.len() as u32) << 24];
    words.extend(args);
    words
}
fn def(n: u32, values: [f32; 4]) -> Vec<u32> {
    instruction(
        81,
        &[vec![0xa00f_0000 | n], values.map(f32::to_bits).to_vec()].concat(),
    )
}

#[test]
fn sparse_inputs_keep_their_declaration_and_a_local_def_is_no_resource() {
    // dcl_texcoord2 v3.yw; mov r0.xy,v3.wywy; mov r0.zw,c0; mov_sat oC0,-abs(r0.zyxw)
    let mut body = instruction(31, &[0x8002_0005, 0x900a_0003]);
    body.extend(instruction(1, &[0x8003_0000, 0x9077_0003]));
    body.extend(instruction(1, &[0x800c_0000, 0xa0e4_0000]));
    body.extend(instruction(1, &[0x800f_0801, 0x80c6_0000]));
    body.extend(instruction(1, &[0x801f_0800, 0x8cc6_0000]));
    let p = lift(&program(&body)).unwrap();
    assert_eq!(p.provenance.format, ShaderFormat::D3d9);
    assert_eq!(p.interface.inputs[0].label.index, 2);
    assert_eq!(p.interface.inputs[0].provenance.d3d.unwrap().mask, 10);
    // def c0 is the program's: no constant bank
    let body = [
        def(0, [1., 2., 3., 4.]),
        instruction(1, &[0x800f_0800, 0xa0ff_0000]),
    ]
    .concat();
    let p = lift(&program(&body)).unwrap();
    assert!(p.interface.resources.is_empty());
}

#[test]
fn cube_and_volume_declarations_keep_their_dimensionality() {
    // Cube and volume declarations retain their dimensionality and lift valid modules.
    for (declaration, dim) in [
        (0x9800_0000, iridia::TextureDim::Cube),
        (0xa000_0000, iridia::TextureDim::D3),
    ] {
        let body = [
            def(0, [0.5; 4]),
            instruction(31, &[declaration, 0xa00f_0800]),
            instruction(66, &[0x800f_0800, 0xa0e4_0000, 0xa0e4_0800]),
        ]
        .concat();
        let p = lift(&program(&body)).unwrap();
        assert!(matches!(p.interface.resources[0].kind,ResourceKind::Texture{dim:d,..} if d==dim));
        iridia::emit::wgsl(&p).unwrap();
    }
}

/// A hand-laid CTAB: header 0..28, constant info 28..48, float4 type 48..64,
/// defaults 64..80, creator/name/target at 80/84/89. One c0, float4 Tint = (1,2,3,4).
fn ctab() -> Vec<u32> {
    let mut b: Vec<u8> = [28u32, 80, 0xffff_0300, 1, 28, 0, 89, 84, 2, 1, 48, 64]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    b.extend([1u16, 3, 1, 4, 1, 0].into_iter().flat_map(u16::to_le_bytes));
    b.extend(0u32.to_le_bytes());
    b.extend(
        [1f32, 2., 3., 4.]
            .into_iter()
            .flat_map(|f| f.to_bits().to_le_bytes()),
    );
    b.extend(b"fxc\0Tint\0ps_3_0\0");
    while !b.len().is_multiple_of(4) {
        b.push(0);
    }
    let mut words = vec![u32::from_le_bytes(*b"CTAB")];
    words.extend(b.as_chunks::<4>().0.iter().map(|s| u32::from_le_bytes(*s)));
    [vec![0xfffe | (words.len() as u32) << 16], words].concat()
}

#[test]
fn constant_table_keeps_names_types_defaults_and_unknown_comments() {
    let body = [
        vec![0x0002_fffe, 0x12345678, 0xabcdef01],
        ctab(),
        instruction(1, &[0x800f_0800, 0xa0e4_0000]),
    ]
    .concat();
    let bytes = program(&body);
    let shader = read(&bytes).unwrap();
    assert_eq!(shader.comments[0].data, [0x12345678, 0xabcdef01]);
    let table = shader.constants.unwrap();
    assert_eq!(table.creator, "fxc");
    assert_eq!(table.target, "ps_3_0");
    let c = &table.constants[0];
    assert_eq!((&*c.name, c.register, c.count), ("Tint", 0, 1));
    assert_eq!(
        (c.ty.class, c.ty.kind, c.ty.rows, c.ty.columns, c.ty.elements),
        (1, 3, 1, 4, 1)
    );
    assert_eq!(
        c.default,
        Some(
            [1f32, 2., 3., 4.]
                .into_iter()
                .flat_map(|f| f.to_bits().to_le_bytes())
                .collect()
        )
    );
    let p = lift(&bytes).unwrap();
    let ResourceKind::ConstantBuffer { layout } = &p.interface.resources[0].kind else {
        panic!()
    };
    assert_eq!(layout.size, 16);
    assert_eq!(layout.constants[0].name, "Tint");
}

#[test]
fn malformed_and_unsupported_programs_are_refused_without_panics() {
    let cases = [
        (instruction(40, &[0xa0e4_0000]), "opcode 40"),
        (instruction(1, &[0x800f_0800, 0x80e4_0000]), "uninitialized"),
        (instruction(1, &[0x800f_0800, 0xa0e4_2000]), "relative"),
        (
            instruction(1 | 1 << 28, &[0x800f_0800, 0xa0e4_0000]),
            "predicate token",
        ),
        (instruction(1, &[0x800f_0800, 0xa2e4_0000]), "modifier"),
        (
            instruction(66, &[0x800f_0800, 0xa0e4_0000, 0xa0e4_0800]),
            "undeclared sampler",
        ),
        (vec![0x0300_0001, 0x800f_0800], "extends past"),
    ];
    for (body, wanted) in cases {
        let err = lift(&program(&body)).unwrap_err().to_string();
        assert!(err.contains(wanted), "{err}");
    }
    let bytes = program(
        &[
            ctab(),
            def(1, [2.; 4]),
            instruction(1, &[0x800f_0800, 0xa0e4_0001]),
        ]
        .concat(),
    );
    for n in 0..bytes.len() {
        let _ = lift(&bytes[..n]);
    }
    for n in 0..bytes.len() {
        for bit in [1, 8, 128] {
            let mut changed = bytes.clone();
            changed[n] ^= bit;
            let _ = lift(&changed);
        }
    }
    assert_eq!(
        read(&[0x00, 0x03, 0xfe, 0xff, 0xff, 0xff, 0, 0]).unwrap().stage,
        iridia::Stage::Vertex
    );
    let mut trailing = bytes;
    trailing.extend([0; 4]);
    assert!(read(&trailing).unwrap_err().detail.contains("after END"));
}

#[test]
fn lifted_fixture_changes_require_a_version_bump() {
    let fixtures = [
        (
            "ctab",
            [ctab(), instruction(1, &[0x800f_0800, 0xa0e4_0000])].concat(),
        ),
        (
            "normalize",
            [
                def(0, [0., 3., 4., 10.]),
                instruction(36, &[0x800f_0000, 0xa0e4_0000]),
                instruction(1, &[0x800f_0800, 0x80e4_0000]),
            ]
            .concat(),
        ),
        (
            "texture",
            [
                def(0, [0.5; 4]),
                instruction(31, &[0x9000_0000, 0xa00f_0800]),
                instruction(66, &[0x800f_0800, 0xa0e4_0000, 0xa0e4_0800]),
            ]
            .concat(),
        ),
        ("relative", instruction(1, &[0x800f_0800, 0xa0e4_2000])),
    ];
    let outcomes = fixtures
        .into_iter()
        .map(|(name, body)| {
            let outcome = lift(&program(&body))
                .map_err(|e| e.to_string())
                .and_then(|p| p.text().map_err(|e| e.to_string()));
            (name.to_owned(), outcome)
        })
        .collect::<Vec<_>>();
    let table = iridia::gate::table("iridia-d3d9", iridia_d3d9::LIFTER_VERSION, &outcomes);
    iridia::gate::check(include_str!("versions.tsv"), &table).unwrap_or_else(|e| panic!("{e}\n{table}"));
}
