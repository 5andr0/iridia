#![allow(clippy::unwrap_used)]
//! The geometry fixture (`geometry.hlsl`: a triangle in, two strips out) in each build: its facts are the
//! ones the entry point states, its function is the one `stage_functions` describes, and it validates and is refused by
//! name by the IR's WGSL writers. Every prefix of it, and of the optimised build, is refused by name, never a panic.
use iridia_dxbc::ir::lift;
use iridia_ir::interface::{GsInput, GsOutput, Stage, StageFacts};
use iridia_ir::target::stage_functions;

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

#[test]
fn a_geometry_program_is_the_function_its_facts_describe() {
    for name in ["geometry.dxbc", "geometry.od.dxbc"] {
        let mut p = lift(&fixture(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        iridia_passes::prepare(&mut p).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(p.stage, Stage::Geometry, "{name}");
        let StageFacts::Geometry(g) = &p.interface.facts else {
            panic!("{name}: {:?}", p.interface.facts)
        };
        assert_eq!(
            (g.input, g.output, g.max_vertices, g.instances),
            (GsInput::Triangle, GsOutput::TriangleStrip, 6, 1),
            "{name}"
        );
        let sigs = stage_functions(&p.interface).unwrap().unwrap();
        let [main] = sigs.as_slice() else { panic!("{name}") };
        let arguments: Vec<_> = main.arguments.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(
            arguments,
            ["SV_Position0", "TEXCOORD0", "SV_PrimitiveID", "SV_GSInstanceID"],
            "{name}"
        );
        let results: Vec<_> = main.result.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(
            results,
            ["SV_Position0", "TEXCOORD0", "SV_VertexCount", "SV_StripStart"],
            "{name}"
        );
        iridia_ir::validate(&p).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        // neither IR writer writes the functions of a stage with no entry point yet (the naga-path writer did): the
        // refusal names the stage, so a writer that learns to write it fails here and this test gets its text check back
        for refusal in [
            iridia_wgsl::wgsl(&p, &iridia_wgsl::Profile::BROWSER)
                .unwrap_err()
                .to_string(),
            iridia_naga::wgsl(&p, &iridia_naga::Profile::NAGA)
                .unwrap_err()
                .to_string(),
        ] {
            assert!(refusal.contains("Geometry program is not"), "{name}: {refusal}");
        }
    }
}

#[test]
fn every_truncation_of_a_geometry_program_is_refused() {
    for name in ["geometry.dxbc", "geometry.od.dxbc"] {
        let bytes = fixture(name);
        for n in 0..bytes.len() {
            assert!(lift(&bytes[..n]).is_err(), "{name}: prefix {n}");
        }
    }
}

struct NoSampler;
impl iridia_exec::Sampler for NoSampler {
    fn sample(&mut self, _: iridia_exec::Sample) -> Result<[f32; 4], String> {
        Err("no sampling in this fixture".into())
    }
}

/// A draw of two triangles through the program in each build, lifted onto the IR and taken through
/// `iridia_passes::prepare`: each triangle emits the two strips the source says (its corners with x scaled, then moved
/// by the primitive number in x and one in y with uv.x one more), and the program as lifted emits the same, bit for bit.
#[test]
fn a_geometry_program_on_the_ir_emits_the_strips_it_states() {
    use iridia_exec::geometry::{GeometryShader, Topology};
    let bits = |x: [f32; 4]| x.map(f32::to_bits);
    let vertices: Vec<Vec<[u32; 4]>> = (0..6)
        .map(|k| {
            let k = k as f32;
            vec![bits([k, 10.0 + k, 0.5, 1.0]), bits([0.25 * k, 0.75, 0.0, 0.0])]
        })
        .collect();
    let mut constants = iridia_exec::Constants::new();
    constants.insert((0, 0), vec![bits([2.0, 0.0, 0.0, 0.0])]);
    for name in ["geometry.dxbc", "geometry.od.dxbc"] {
        let bytes = fixture(name);
        let ir = iridia_dxbc::ir::lift(&bytes).unwrap_or_else(|e| panic!("{name} on the IR: {e}"));
        let mut prepared = ir.clone();
        iridia_passes::prepare(&mut prepared).unwrap_or_else(|e| panic!("{name}: {e}"));
        let a = GeometryShader::from_ir(&prepared).unwrap_or_else(|e| panic!("{name}: {e}"));
        let b = GeometryShader::from_ir(&ir).unwrap_or_else(|e| panic!("{name} as lifted: {e}"));
        let draw = |s: &GeometryShader| {
            s.draw(&constants, &mut NoSampler, &vertices, Topology::TriangleList)
                .unwrap_or_else(|e| panic!("{name}: {e}"))
        };
        let (strips, ir_strips) = (draw(&a), draw(&b));
        assert_eq!(ir_strips, strips, "{name}: the strips of the program as lifted");
        assert_eq!(strips.len(), 4, "{name}: two strips a triangle");
        for (k, strip) in strips.iter().enumerate() {
            let primitive = (k / 2) as u32;
            assert_eq!(
                (strip.primitive, strip.instance),
                (primitive, 0),
                "{name} strip {k}"
            );
            for (corner, v) in strip.vertices.iter().enumerate() {
                let input = &vertices[primitive as usize * 3 + corner];
                let [x, y, z, w] = input[0].map(f32::from_bits);
                let [u, t, ..] = input[1].map(f32::from_bits);
                let (position, uv) = if k % 2 == 0 {
                    ([x * 2.0, y, z, w], [u, t])
                } else {
                    ([x + primitive as f32, y + 1.0, z, w], [u + 1.0, t])
                };
                assert_eq!(v[0], bits(position), "{name} strip {k} vertex {corner}: position");
                assert_eq!(
                    &v[1][..2],
                    &bits([uv[0], uv[1], 0.0, 0.0])[..2],
                    "{name} strip {k} vertex {corner}: uv"
                );
            }
        }
    }
}
