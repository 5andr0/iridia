#![allow(clippy::unwrap_used)]
//! Early depth must suppress the shader's UAV writes when depth rejects a covered pixel. The otherwise identical
//! program without the attribute still executes its writes before the failed depth test.
use iridia_exec::draw::{Compare, Cull, DepthState, Primitive, Rasterizer, ShadedDraw, Surface, Viewport};
use iridia_exec::geometry::Topology;
use iridia_exec::{ProgramShader, Sample, Sampler, Store};
use iridia_ir::interface::{Label, ScalarType, SystemValue, Value, Varying, VaryingProvenance};

#[derive(Default)]
struct Marks(Vec<u32>);
impl Sampler for Marks {
    fn sample(&mut self, _: Sample) -> Result<[f32; 4], String> {
        Err("the shader has no textures".into())
    }
    fn store(&mut self, request: Store) -> Result<(), String> {
        assert_eq!(request.view, 1);
        assert_eq!(request.value[0], 1);
        self.0.push(request.address[0]);
        Ok(())
    }
    fn buffer_elements(&mut self, _: u32) -> Option<u32> {
        Some(2)
    }
}

fn draw(program: iridia_ir::Program, compare: Compare, depths: [f32; 4]) -> (Marks, Surface) {
    let shader = ProgramShader::from_ir(program).unwrap();
    let outputs = vec![Varying {
        label: Label {
            name: "SV_Position".into(),
            index: 0,
        },
        system: Some(SystemValue::ClipPosition),
        value: Value {
            scalar: ScalarType::Float,
            width: 4,
        },
        interp: None,
        used: true,
        provenance: VaryingProvenance::default(),
    }];
    let draw = ShadedDraw::new(&outputs, &shader).unwrap();
    let vertices = [
        [-1.0f32, -1.0, 0.5, 1.0],
        [3.0, -1.0, 0.5, 1.0],
        [-1.0, 3.0, 0.5, 1.0],
    ]
    .map(|p| vec![p.map(f32::to_bits)]);
    let state = Rasterizer {
        viewports: vec![Viewport {
            x: 0.0,
            y: 0.0,
            width: 2.0,
            height: 2.0,
            min_depth: 0.0,
            max_depth: 1.0,
        }],
        sample_mask: u32::MAX,
        cull: Cull::None,
        front_counter_clockwise: false,
        depth: Some(DepthState {
            compare,
            write: false,
            bounds: None,
        }),
        alpha_test: None,
        clip_planes: [None; 6],
        point_size: None,
        point_sprite: false,
        flat_colours: false,
        clamp_colours: false,
        fog: None,
    };
    let mut surface = Surface {
        width: 2,
        height: 2,
        layers: 1,
        samples: 1,
        targets: vec![Some(vec![[0; 4]; 4])],
        depth: Some(depths.to_vec()),
    };
    let mut marks = Marks::default();
    draw.draw(
        &vertices,
        &Primitive::assemble(Topology::TriangleList, &[0, 1, 2], None).unwrap(),
        &[],
        &mut marks,
        &state,
        &mut surface,
    )
    .unwrap();
    (marks, surface)
}

#[test]
fn early_depth_suppresses_uav_writes_on_rejected_pixels() {
    for (family, bytes, early) in [
        (
            "dxbc",
            &include_bytes!("../../iridia-dxbc/tests/fixtures/early_depth.dxbc")[..],
            true,
        ),
        (
            "dxbc",
            &include_bytes!("../../iridia-dxbc/tests/fixtures/late_depth.dxbc")[..],
            false,
        ),
        (
            "dxil",
            &include_bytes!("../../iridia-dxil/tests/fixtures/early_depth.dxil")[..],
            true,
        ),
        (
            "dxil",
            &include_bytes!("../../iridia-dxil/tests/fixtures/late_depth.dxil")[..],
            false,
        ),
    ] {
        let mut p = if family == "dxbc" {
            iridia_dxbc::ir::lift(bytes)
        } else {
            iridia_dxil::ir::lift(bytes)
        }
        .unwrap();
        iridia_passes::prepare(&mut p).unwrap();
        for (compare, depths, visible) in [
            (Compare::Never, [1.0; 4], [false; 4]),
            (Compare::Always, [1.0; 4], [true; 4]),
            (Compare::Less, [0.25, 1.0, 0.25, 1.0], [false, true, false, true]),
        ] {
            let (mut marks, surface) = draw(p.clone(), compare, depths);
            marks.0.sort();
            let mut expected: Vec<u32> = if early {
                visible
                    .iter()
                    .enumerate()
                    .filter(|(_, shown)| **shown)
                    .map(|(i, _)| u32::try_from(i % 2).unwrap())
                    .collect()
            } else {
                vec![0, 0, 1, 1]
            };
            expected.sort();
            assert_eq!(marks.0, expected, "{family}: early={early}, {compare:?}");
            for (pixel, shown) in surface.targets[0].as_ref().unwrap().iter().zip(visible) {
                if shown {
                    assert_eq!(pixel[3], 1.0f32.to_bits());
                } else {
                    assert_eq!(*pixel, [0; 4]);
                }
            }
        }
    }
}
