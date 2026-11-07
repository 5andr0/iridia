#![allow(clippy::unwrap_used)]
//! The mesh fixture (`mesh.hlsl`: a quad as two triangles, a thread group of 32) in each build: its facts are the ones the
//! entry point states, it is a mesh entry point with D3D's thread group and the group index as its one argument, it
//! validates and is written as WGSL with the mesh stage's attributes, and its IR lift draws the quad the HLSL writes.
use iridia::{MeshFacts, MeshTopology, Stage, StageFacts};
use iridia_dxil::lift;

#[test]
fn a_mesh_program_is_a_mesh_entry_point_with_its_facts() {
    for name in ["mesh.dxil", "mesh.od.dxil", "mesh.66.dxil"] {
        let bytes = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures")
                .join(name),
        )
        .unwrap();
        let p = lift(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(p.stage, Stage::Mesh, "{name}");
        assert_eq!(
            p.interface.facts,
            StageFacts::Mesh(MeshFacts {
                topology: MeshTopology::Triangle,
                max_vertices: 4,
                max_primitives: 2,
                payload: 0,
            }),
            "{name}"
        );
        let [entry] = p.module.entry_points.as_slice() else {
            panic!("{name}")
        };
        assert_eq!(entry.stage, naga::ShaderStage::Mesh, "{name}");
        assert_eq!(entry.workgroup_size, [32, 1, 1], "{name}");
        let arguments: Vec<_> = entry
            .function
            .arguments
            .iter()
            .map(|a| a.name.as_deref().unwrap_or(""))
            .collect();
        assert_eq!(arguments, ["SV_GroupIndex0"], "{name}");
        let outputs: Vec<_> = p
            .interface
            .outputs
            .iter()
            .map(|v| v.label.name.as_str())
            .collect();
        assert_eq!(outputs, ["SV_Position", "COLOR"], "{name}");
        p.validate().unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let wgsl = iridia::emit::wgsl(&p).unwrap();
        assert!(wgsl.contains("@mesh(mesh_output)"), "{name}: {wgsl}");
        assert!(wgsl.contains("@workgroup_size(32, 1, 1)"), "{name}");
        assert!(wgsl.contains("@builtin(triangle_indices)"), "{name}");
    }
}

fn nothing(_: iridia_exec::Sample) -> Result<[f32; 4], String> {
    Err("no sampling in this program".into())
}

/// Each build lifted onto the IR ([`iridia_dxil::ir`]) draws the quad `mesh.hlsl` writes, in every thread group of a
/// dispatch: four vertices, vertex i at the corner (i & 1, i >> 1) scaled by b0's `scale` and coloured by its corner, and
/// the triangles (0, 1, 2) and (2, 1, 3).
#[test]
fn a_mesh_program_on_the_ir_draws_the_quad_its_hlsl_writes() {
    for name in ["mesh.dxil", "mesh.od.dxil", "mesh.66.dxil"] {
        let bytes = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures")
                .join(name),
        )
        .unwrap();
        let mut p = iridia_dxil::ir::lift(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(p.workgroup, Some([32, 1, 1]), "{name}");
        iridia_passes::prepare(&mut p).unwrap();
        let b = iridia_exec::mesh::MeshShader::from_ir(&p).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            b.facts(),
            MeshFacts {
                topology: MeshTopology::Triangle,
                max_vertices: 4,
                max_primitives: 2,
                payload: 0,
            },
            "{name}"
        );
        assert_eq!(
            b.outputs().collect::<Vec<_>>(),
            ["SV_Position0", "COLOR0"],
            "{name}"
        );
        let scale = 1.5f32;
        let constants = [((0, 0), vec![[scale.to_bits(), 0, 0, 0]])].into_iter().collect();
        let got = b.draw(&constants, &mut nothing, [3, 2, 1], 32).unwrap();
        assert_eq!(got.len(), 6, "{name}: a meshlet per thread group");
        for (k, meshlet) in got.iter().enumerate() {
            let k = k as u32;
            assert_eq!(meshlet.group, [k % 3, k / 3, 0], "{name}");
            assert_eq!(meshlet.primitives, [vec![0, 1, 2], vec![2, 1, 3]], "{name}");
            assert_eq!(meshlet.vertices.len(), 4, "{name}");
            for (i, vertex) in (0u32..).zip(&meshlet.vertices) {
                let corner = [(i & 1) as f32, (i >> 1) as f32];
                let position = [corner[0] * scale, corner[1] * scale, 0.0, 1.0].map(f32::to_bits);
                assert_eq!(vertex[0], position, "{name}: group {k}, vertex {i}'s position");
                assert_eq!(
                    vertex[1][..3],
                    [corner[0], corner[1], 0.0].map(f32::to_bits),
                    "{name}: group {k}, vertex {i}'s colour"
                );
            }
        }
    }
}

/// The meshlet stores in the IR's text form, read back as they were printed; a program of another stage writing a
/// meshlet is refused by the IR's validator.
#[test]
fn meshlet_stores_round_trip_and_belong_to_mesh_programs() {
    let bytes = std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join("mesh.dxil"),
    )
    .unwrap();
    let p = iridia_dxil::ir::lift(&bytes).unwrap();
    let text = iridia_ir::text::print(&p.entry);
    for line in ["mesh counts %", "store vertex %", "store indices %"] {
        assert!(text.contains(line), "{line}\n{text}");
    }
    let again = iridia_ir::text::print(&iridia_ir::text::parse(&text).unwrap());
    assert_eq!(again, text);
    let mut q = p.clone();
    q.stage = Stage::Compute;
    q.interface.stage = Stage::Compute;
    q.interface.facts = StageFacts::None;
    let problems = iridia_ir::validate(&q).unwrap_err();
    assert!(
        problems
            .iter()
            .any(|e| e.to_string().contains("a meshlet written outside a mesh program")),
        "{problems:?}"
    );
}
