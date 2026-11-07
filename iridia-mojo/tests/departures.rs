#![allow(clippy::unwrap_used)]
//! Where MojoShader's GLSL is not its Direct3D 9 program (`pairing::departures`, which names such a pair by the
//! instruction), each shown on a program of three or four instructions, hand-assembled here: the program lifted by
//! `iridia-d3d9-ir` computes what Direct3D 9's instruction says, the GLSL lifted by `iridia_mojo::ir` what its own text
//! says, each checked against a plain Rust model of its meaning, and the two differ.
mod pairing;

use iridia_ir::interface::{Interface, ResourceId, ResourceKind};
use pairing::{AttributeLine, HintsLine, Line, UniformLine};

/// The lift pairing's D has: IEEE 754's multiply, which GLSL's is.
fn options() -> iridia_d3d9_ir::Options {
    iridia_d3d9_ir::Options {
        multiply: iridia_ir::MulZero::Ieee,
        ..Default::default()
    }
}

/// The words of a program as the bytes of a shader blob, ending with `END`.
fn bytes(words: &[u32]) -> Vec<u8> {
    words
        .iter()
        .chain(&[0x0000_FFFF])
        .flat_map(|w| w.to_le_bytes())
        .collect()
}

fn line(name: &str, bytes: &[u8], glsl: &str, stage: &str, target: &str, uniforms: Vec<UniformLine>) -> Line {
    let sha256 = {
        use sha2::{Digest, Sha256};
        Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
    };
    Line {
        name: name.into(),
        sha256,
        refused: None,
        glsl: Some(glsl.into()),
        stage: Some(stage.into()),
        target: Some(target.into()),
        hints: None,
        uniforms,
    }
}

/// `nrm r0, v1` of a four-lane register: Direct3D 9 divides by the length of x, y and z; MojoShader's
/// `normalize(vs_v1)` is the length of the four lanes.
#[test]
fn nrm_of_four_lanes_takes_three_lanes_in_direct3d_and_four_in_the_glsl() {
    let program = bytes(&[
        0xFFFE_0300, // vs_3_0
        0x0200_001F,
        0x8000_0000,
        0x900F_0000, // dcl_position v0
        0x0200_001F,
        0x8000_0005,
        0x900F_0001, // dcl_texcoord v1
        0x0200_001F,
        0x8000_0000,
        0xE00F_0000, // dcl_position o0
        0x0200_001F,
        0x8000_0005,
        0xE00F_0001, // dcl_texcoord o1
        0x0200_0001,
        0xE00F_0000,
        0x90E4_0000, // mov o0, v0
        0x0200_0024,
        0x800F_0000,
        0x90E4_0001, // nrm r0, v1
        0x0200_0001,
        0xE00F_0001,
        0x80E4_0000, // mov o1, r0
    ]);
    let glsl = "uniform vec4 posFixup;\nvec4 vs_r0;\nattribute vec4 vs_v0;\n#define vs_o0 gl_Position\n\
attribute vec4 vs_v1;\nvarying vec4 vTexCoord0;\n#define vs_o1 vTexCoord0\n\nvoid main()\n{\n\
\tvs_o0 = vs_v0;\n\tvs_r0 = normalize(vs_v1);\n\tvs_o1 = vs_r0;\n\
\tgl_Position.y = gl_Position.y * posFixup.y;\n\tgl_Position.xy += posFixup.zw * gl_Position.ww;\n\
\tgl_Position.z = gl_Position.z * 2.0 - gl_Position.w;\n}\n";
    let mut pair = line("nrm4.vs", &program, glsl, "vertex", "vs_3_0", Vec::new());
    pair.hints = Some(HintsLine {
        constants: Vec::new(),
        samplers: Vec::new(),
        attributes: [("vs_v0", "POSITION"), ("vs_v1", "TEXCOORD")]
            .map(|(glsl_name, semantic)| AttributeLine {
                glsl_name: glsl_name.into(),
                semantic: semantic.into(),
                index: 0,
            })
            .into(),
    });
    let hints = pair.hints();
    let d = iridia_d3d9_ir::lift_with(&program, &options()).unwrap();
    let m = iridia_mojo::ir::lift(glsl, pair.stage(), &hints).unwrap();
    let inputs = |interface: &Interface, v: [f32; 4]| -> Vec<[u32; 4]> {
        interface
            .inputs
            .iter()
            .map(|i| {
                if i.label.name == "TEXCOORD" {
                    v.map(f32::to_bits)
                } else {
                    [0; 4]
                }
            })
            .collect()
    };
    let mut differing = 0;
    for v in [
        [3.0, 4.0, 12.0, 0.0],
        [3.0, 4.0, 12.0, 5.0],
        [-0.25, 1.5, 0.75, -2.0],
        [1.0e-3, -2.0e-3, 4.0e-3, 9.0e-3],
        [100.0, -200.0, 300.0, 400.0],
        [0.5, 0.0, 0.0, 1.0],
    ] {
        let d_out = pairing::ir_views(&d, &[inputs(&d.interface, v)], &[]).unwrap()[0].varying("TEXCOORD", 0);
        let m_out = pairing::ir_views(&m, &[inputs(&m.interface, v)], &[]).unwrap()[0].varying("TEXCOORD", 0);
        let v64 = v.map(f64::from);
        let length = |n: usize| v64[..n].iter().map(|x| x * x).sum::<f64>().sqrt();
        for (got, n, which) in [
            (d_out, 3, "D, the length of x, y and z"),
            (m_out, 4, "M, the length of x, y, z and w"),
        ] {
            for l in 0..4 {
                let want = (v64[l] / length(n)) as f32;
                let got = f32::from_bits(got[l]);
                assert!(
                    (got - want).abs() <= 4.0 * f32::EPSILON * want.abs().max(f32::MIN_POSITIVE),
                    "{v:?} lane {l}: {which} wants {want}, has {got}"
                );
            }
        }
        differing += usize::from(d_out != m_out);
    }
    assert!(differing >= 5, "{differing} of the inputs tell the lengths apart");
    let (class, _) = pairing::pair("nrm4.vs", &program, &pair, 8);
    assert_eq!(class, format!("{}{}", pairing::DEPARTS, pairing::NRM));
}

/// `cmp r0, c0, r0.z, r0.w`: Direct3D 9 reads the old r0 for every lane, MojoShader's `ps_r0.z = ...; ps_r0.w = ...
/// ps_r0.z ...` the lane the line before wrote.
#[test]
fn cmp_reads_the_old_register_in_direct3d_and_the_new_lane_in_the_glsl() {
    let program = bytes(&[
        0xFFFF_0200, // ps_2_0
        0x0500_0051,
        0xA00F_0001,
        0x0000_0000,
        0x3F80_0000,
        0x8000_0000,
        0xBF80_0000, // def c1, 0, 1, -0, -1
        0x0200_0001,
        0x800F_0000,
        0xA0E4_0001, // mov r0, c1
        0x0400_0058,
        0x800F_0001,
        0xA1E4_0000,
        0x8000_0000,
        0x8055_0000, // cmp r1, -c0, r0.x, r0.y
        0x0400_0058,
        0x800F_0000,
        0xA0E4_0000,
        0x80AA_0000,
        0x80FF_0000, // cmp r0, c0, r0.z, r0.w
        0x0300_0002,
        0x800F_0000,
        0x80E4_0000,
        0x80E4_0001, // add r0, r0, r1
        0x0200_0001,
        0x800F_0800,
        0x80E4_0000, // mov oC0, r0
    ]);
    let glsl = "uniform vec4 ps_uniforms_vec4[1];\nconst vec4 ps_c1 = vec4(0.0, 1.0, -0.0, -1.0);\nvec4 ps_r0;\nvec4 ps_r1;\n\
#define ps_c0 ps_uniforms_vec4[0]\n#define ps_oC0 gl_FragColor\n\nvoid main()\n{\n\tps_r0 = ps_c1;\n\
\tps_r1.x = ((-ps_c0.x >= 0.0) ? ps_r0.x : ps_r0.y);\n\tps_r1.y = ((-ps_c0.y >= 0.0) ? ps_r0.x : ps_r0.y);\n\
\tps_r1.z = ((-ps_c0.z >= 0.0) ? ps_r0.x : ps_r0.y);\n\tps_r1.w = ((-ps_c0.w >= 0.0) ? ps_r0.x : ps_r0.y);\n\
\tps_r0.x = ((ps_c0.x >= 0.0) ? ps_r0.z : ps_r0.w);\n\tps_r0.y = ((ps_c0.y >= 0.0) ? ps_r0.z : ps_r0.w);\n\
\tps_r0.z = ((ps_c0.z >= 0.0) ? ps_r0.z : ps_r0.w);\n\tps_r0.w = ((ps_c0.w >= 0.0) ? ps_r0.z : ps_r0.w);\n\
\tps_r0 = ps_r0 + ps_r1;\n\tps_oC0 = ps_r0;\n}\n";
    let uniforms = vec![UniformLine {
        kind: "float".into(),
        register: 0,
        array_count: 1,
        constant: false,
    }];
    let pair = line("cmp.ps", &program, glsl, "pixel", "ps_2_0", uniforms);
    let d = iridia_d3d9_ir::lift_with(&program, &options()).unwrap();
    let m = iridia_mojo::ir::lift(glsl, pair.stage(), &pair.hints()).unwrap();
    let bank = |p: &iridia_ir::Program, c0: [f32; 4]| -> Vec<(ResourceId, u32, Vec<[u32; 4]>)> {
        p.interface
            .resources
            .iter()
            .filter_map(|r| match (&r.kind, r.provenance.d3d) {
                (ResourceKind::ConstantBuffer { layout }, Some(b)) if b.register == 0 => {
                    let mut rows = vec![[0u32; 4]; layout.size.div_ceil(16) as usize];
                    rows[0] = c0.map(f32::to_bits);
                    Some((r.id, 0, rows))
                }
                _ => None,
            })
            .collect()
    };
    let (mut differing, mut cases) = (0, 0);
    let values = [-2.0f32, -0.0, 0.0, 1.5, -1.5];
    for a in values {
        for b in values {
            for c in values {
                for e in values {
                    let c0 = [a, b, c, e];
                    // the program, by its text: `old` is r0 as every lane reads it
                    let old = [0.0f32, 1.0, -0.0, -1.0];
                    let r1: Vec<f32> = c0
                        .iter()
                        .map(|&x| if -x >= 0.0 { old[0] } else { old[1] })
                        .collect();
                    let parallel: Vec<f32> = (0..4)
                        .map(|k| (if c0[k] >= 0.0 { old[2] } else { old[3] }) + r1[k])
                        .collect();
                    let mut r0 = old;
                    for k in 0..4 {
                        r0[k] = if c0[k] >= 0.0 { r0[2] } else { r0[3] };
                    }
                    let sequential: Vec<f32> = (0..4).map(|k| r0[k] + r1[k]).collect();
                    let of = |p: &iridia_ir::Program| {
                        pairing::ir_views(p, &[Vec::new(), Vec::new(), Vec::new(), Vec::new()], &bank(p, c0))
                            .unwrap()[0]
                            .target(0)
                    };
                    let (d_out, m_out) = (of(&d), of(&m));
                    let bits = |v: &[f32]| -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() };
                    assert_eq!(d_out.to_vec(), bits(&parallel), "{c0:?}: D");
                    assert_eq!(m_out.to_vec(), bits(&sequential), "{c0:?}: M");
                    differing += usize::from(d_out != m_out);
                    cases += 1;
                }
            }
        }
    }
    assert!(
        differing > 0 && differing < cases,
        "{differing} of {cases} constants tell them apart"
    );
    let (class, _) = pairing::pair("cmp.ps", &program, &pair, 64);
    assert_eq!(class, format!("{}{}", pairing::DEPARTS, pairing::CMP));
}
