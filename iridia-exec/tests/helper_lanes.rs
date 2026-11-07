#![allow(clippy::unwrap_used)]
//! Raster helpers compute derivatives but never write the caller's UAVs.
mod support;
use iridia_exec::draw::{Compare, Cull, DepthState, Primitive, Rasterizer, ShadedDraw, Surface, Viewport};
use iridia_exec::geometry::Topology;
use iridia_exec::{ProgramShader, Sample, Sampler, Store};
use iridia_ir::interface::{
    Access, BufferKind, D3dClass, ResourceKind, ScalarType, StorageTarget, SystemValue,
};
use iridia_ir::interface::{Label, Value, Varying, VaryingProvenance};
use support::{pixel, resource, target, varying};
#[derive(Default)]
struct Marks(Vec<u32>);
impl Sampler for Marks {
    fn sample(&mut self, _: Sample) -> Result<[f32; 4], String> {
        Err("the shader has no textures".into())
    }
    fn store(&mut self, request: Store) -> Result<(), String> {
        assert_eq!(request.view, 0);
        assert_eq!(request.value[0], 1.0f32.to_bits());
        self.0.push(request.address[0]);
        Ok(())
    }
    fn buffer_elements(&mut self, _: u32) -> Option<u32> {
        Some(2)
    }
}

fn draw(program: iridia_ir::Program) -> (Marks, Surface) {
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
            compare: Compare::Always,
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
        depth: Some(vec![1.0; 4]),
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
fn fullscreen_triangle_writes_once_per_pixel_and_helpers_feed_derivatives() {
    let program = pixel(
        "iridia-ir 1
fn main {
  var private: f32
  %0: vec4<f32> = input 0.xyzw
  %1: f32 = %0.x
  %2: f32 = %0.y
  private = %1
  %3: f32 = load private
  %4: f32 = deriv.x.coarse %3
  %5: f32 = deriv.y.coarse %2
  %6: vec4<f32> = compose %4, %5, %4, %5
  store buffer r0 + 0 = %4
  output 0.xyzw = %6
  return
}
",
        vec![varying(
            "SV_Position",
            0,
            ScalarType::Float,
            4,
            Some((0, 15)),
            Some(SystemValue::FragCoord),
        )],
        vec![target()],
        vec![resource(
            0,
            "marks",
            ResourceKind::Storage {
                target: StorageTarget::Buffer(BufferKind::Raw),
                access: Access::ReadWrite,
            },
            D3dClass::UnorderedAccess,
            0,
            1,
        )],
    );
    let (marks, surface) = draw(program);
    assert_eq!(marks.0, vec![0; 4], "helper lanes must perform no UAV stores");
    assert_eq!(
        surface.targets[0].as_ref().unwrap(),
        &vec![[1.0f32.to_bits(); 4]; 4]
    );
}
