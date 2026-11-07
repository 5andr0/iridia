#![allow(clippy::unwrap_used)]
//! The program executor on programs of the IR written as text: a scalar run with the derivatives the caller states, a
//! quad's own derivatives, the requests a 3D sample makes of its sampler, resource arrays, variables, and the float
//! operations. Where the reference interpreter (`iridia-exec-ir`) runs the program too, the two agree.
mod support;

use iridia_exec::probe::AnsweredReads;
use iridia_exec::{Dim, ImageLod, ImageSample, PixelShader, ProgramShader, Sampler};
use iridia_ir::Program;
use iridia_ir::interface::{BufferLayout, D3dClass, Resource, ResourceKind, SampleType, TextureDim};
use support::{agree, input, pixel, reference, reference_quad, resource, target};

/// `SV_Target0 = float4(uv.x * cb0[0].x, uv.y * cb0[0].y, ddx(uv.x), ddy(uv.y))`, the constant buffer at `cb3`.
fn derivative_program() -> Program {
    let text = "iridia-ir 1
fn main {
  %0: vec2<f32> = input 0.xy
  %1: vec4<u32> = buffer r0 + 0 oob.zero
  %2: vec4<f32> = bitcast %1
  %3: vec2<f32> = %2.xy
  %4: vec2<f32> = fmul.ieee %0, %3
  %5: f32 = %0.x
  %6: f32 = %0.y
  %7: f32 = deriv.x.coarse %5
  %8: f32 = deriv.y.coarse %6
  %9: f32 = %4.x
  %10: f32 = %4.y
  %11: vec4<f32> = compose %9, %10, %7, %8
  output 0.xyzw = %11
  return
}
";
    let cb = resource(
        0,
        "cb0",
        ResourceKind::ConstantBuffer {
            layout: BufferLayout {
                size: 16,
                constants: vec![],
            },
        },
        D3dClass::ConstantBuffer,
        3,
        1,
    );
    pixel(text, vec![input("TEXCOORD", 2, 0x3)], vec![target()], vec![cb])
}

/// `SV_Target0 = tex3d.Sample(smp, TEXCOORD0.xyz)`, the volume at `t2` and the sampler at `s1`.
fn volume_program() -> Program {
    let text = "iridia-ir 1
fn main {
  %0: vec3<f32> = input 0.xyz
  %1: vec4<f32> = sample r0, r1, %0
  output 0.xyzw = %1
  return
}
";
    let volume = resource(
        0,
        "volume",
        ResourceKind::Texture {
            dim: TextureDim::D3,
            arrayed: false,
            multisampled: false,
            sample: SampleType::Float,
        },
        D3dClass::ShaderResource,
        2,
        1,
    );
    let linear = resource(
        1,
        "linear",
        ResourceKind::Sampler { comparison: false },
        D3dClass::Sampler,
        1,
        1,
    );
    pixel(
        text,
        vec![input("TEXCOORD", 3, 0x7)],
        vec![target()],
        vec![volume, linear],
    )
}

struct Unused;
impl Sampler for Unused {
    fn sample(&mut self, _: iridia_exec::Sample) -> Result<[f32; 4], String> {
        Err("no samples expected".into())
    }
}

fn no_samples(_: iridia_exec_ir::Sample) -> Result<[f32; 4], String> {
    Err("no samples expected".into())
}

fn bits(x: [f32; 4]) -> [u32; 4] {
    x.map(f32::to_bits)
}

fn lanes(x: [u32; 4]) -> [f32; 4] {
    x.map(f32::from_bits)
}

#[test]
fn a_scalar_run_reads_the_derivatives_the_caller_states() {
    let shader = ProgramShader::from_ir(derivative_program()).unwrap();
    assert!(shader.needs_quad());
    assert!(!ProgramShader::from_ir(branch_program()).unwrap().needs_quad());
    assert_eq!(shader.input_count(), 1);
    assert_eq!(shader.constant_buffers(), &[(3, 1)]);
    let cb = [bits([2.0, 3.0, 0.0, 0.0])];
    let inputs = [bits([0.25, 0.5, 0.0, 0.0])];
    let derivatives = [[bits([0.125, 0.0, 0.0, 0.0]), bits([0.0, 0.0625, 0.0, 0.0])]];
    let e = shader
        .execute_with_derivatives(&inputs, &derivatives, &[(3, &cb)], &mut Unused)
        .unwrap();
    assert_eq!(lanes(e.outputs[0]), [0.5, 1.5, 0.125, 0.0625]);
    assert_eq!(e.output_masks, vec![0xf]);
    assert_eq!(e.derivatives, 2);
    // the derivatives of an input are an error where they are not given
    let e = shader.execute(&inputs, &[(3, &cb)], &mut Unused).unwrap_err();
    assert!(e.detail.contains("derivatives"), "{e}");
    // a missing constant buffer
    let e = shader.execute(&inputs, &[], &mut Unused).unwrap_err();
    assert!(e.detail.contains("cb3"), "{e}");
}

#[test]
fn a_quad_takes_its_own_derivatives_from_its_four_lanes() {
    let program = derivative_program();
    let shader = ProgramShader::from_ir(program.clone()).unwrap();
    let cb = [bits([2.0, 3.0, 0.0, 0.0])];
    // lanes 0, 1 the top row, 2, 3 the bottom: x grows 0.125 to the right, y 0.0625 downwards
    let lane = |k: u32| {
        [bits([
            0.25 + (k & 1) as f32 * 0.125,
            0.5 + (k >> 1) as f32 * 0.0625,
            0.0,
            0.0,
        ])]
    };
    let inputs = [lane(0), lane(1), lane(2), lane(3)];
    let e = shader
        .execute_quad(
            [&inputs[0], &inputs[1], &inputs[2], &inputs[3]],
            &[(3, &cb)],
            &mut Unused,
        )
        .unwrap();
    let reference = reference_quad(
        &program,
        [&inputs[0], &inputs[1], &inputs[2], &inputs[3]],
        &[(3, &cb)],
        &no_samples,
    );
    for (k, (e, run)) in e.iter().zip(&reference).enumerate() {
        let (x, y) = (0.25 + (k & 1) as f32 * 0.125, 0.5 + (k >> 1) as f32 * 0.0625);
        assert_eq!(lanes(e.outputs[0]), [x * 2.0, y * 3.0, 0.125, 0.0625], "lane {k}");
        // each lane takes part in the program's two derivatives
        assert_eq!(e.derivatives, 2, "lane {k}");
        agree(&program, e, run);
    }
}

#[test]
fn a_volume_sample_reaches_a_sampler_that_answers_it_and_is_refused_by_one_that_does_not() {
    let program = volume_program();
    let shader = ProgramShader::from_ir(program.clone()).unwrap();
    // a volume is reached through sample_image: textures() lists the planes Sampler::sample answers
    assert_eq!(shader.textures(), &[] as &[u32]);
    assert_eq!(shader.samplers(), &[1]);
    let inputs = [bits([0.1, 0.2, 0.3, 0.0])];
    let mut sampler = AnsweredReads::new(|_: &ImageSample<u32>| Ok([0.5, 0.25, 0.125, 1.0]));
    let e = shader.execute(&inputs, &[], &mut sampler).unwrap();
    assert_eq!(lanes(e.outputs[0]), [0.5, 0.25, 0.125, 1.0]);
    assert_eq!(e.samples, 1);
    let [r] = sampler.samples.as_slice() else {
        panic!("{:?}", sampler.samples)
    };
    assert_eq!((r.texture, r.sampler, r.dim), (2, 1, Dim::D3));
    assert_eq!(r.coord, [0.1, 0.2, 0.3]);
    assert_eq!(r.lod, ImageLod::Implicit);
    assert!(r.offset == [0; 3] && r.gather.is_none() && r.depth_ref.is_none() && !r.depth);
    // the reference asks its sampler the same: the volume and the sampler, the three lanes, the implicit level
    let mut asked = Vec::new();
    let run = reference(&program, &inputs, &[], &mut |s| {
        asked.push(s);
        Ok([0.5, 0.25, 0.125, 1.0])
    });
    agree(&program, &e, &run);
    let [s] = asked.as_slice() else {
        panic!("{asked:?}")
    };
    assert_eq!(
        (s.texture.0, s.sampler.0, s.coord, s.lanes, s.lod),
        (0, 1, [0.1, 0.2, 0.3], 3, iridia_exec_ir::Lod::Auto)
    );
    // the default sampler answers plain 2D samples only
    let e = shader.execute(&inputs, &[], &mut Unused).unwrap_err();
    assert!(e.detail.contains("plain 2D samples only"), "{e}");
}

/// `if (uv.x > 0.5) { return red; } return green;`: two returns, each after storing a result of its own.
fn branch_program() -> Program {
    let text = "iridia-ir 1
fn main {
  %0: vec2<f32> = input 0.xy
  %1: f32 = %0.x
  %2: f32 = f32(0.5)
  %3: bool = fcmp.gt %1, %2
  if %3 {
    %4: vec4<f32> = vec4<f32>(1.0, 0.0, 0.0, 1.0)
    output 0.xyzw = %4
    return
  } else {
    yield
  }
  %5: vec4<f32> = vec4<f32>(0.0, 1.0, 0.0, 1.0)
  output 0.xyzw = %5
  return
}
";
    pixel(text, vec![input("TEXCOORD", 2, 0x3)], vec![target()], Vec::new())
}

#[test]
fn each_return_leaves_with_its_own_result() {
    let program = branch_program();
    let shader = ProgramShader::from_ir(program.clone()).unwrap();
    let run = |x: f32| {
        let inputs = [bits([x, 0.0, 0.0, 0.0])];
        let e = shader.execute(&inputs, &[], &mut Unused).unwrap();
        agree(&program, &e, &reference(&program, &inputs, &[], &mut no_samples));
        lanes(e.outputs[0])
    };
    assert_eq!(run(0.75), [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(run(0.25), [0.0, 1.0, 0.0, 1.0]);
    // a quad whose lanes leave at different returns
    let lane = |x: f32| vec![bits([x, 0.0, 0.0, 0.0])];
    let inputs = [lane(0.25), lane(0.75), lane(0.25), lane(0.75)];
    let e = shader
        .execute_quad([&inputs[0], &inputs[1], &inputs[2], &inputs[3]], &[], &mut Unused)
        .unwrap();
    let colours: Vec<[f32; 4]> = e.iter().map(|e| lanes(e.outputs[0])).collect();
    assert_eq!(
        colours,
        [
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0]
        ]
    );
}

/// `SV_Target0 = float4(uv.x, uv.y, uv.y, uv.x)`, the input read by two operations: both are the one input.
#[test]
fn an_input_read_by_several_operations_is_the_same_value_in_each() {
    let text = "iridia-ir 1
fn main {
  %0: vec2<f32> = input 0.xy
  %1: vec2<f32> = input 0.xy
  %2: f32 = %0.x
  %3: f32 = %1.y
  %4: vec4<f32> = compose %2, %3, %3, %2
  output 0.xyzw = %4
  return
}
";
    let program = pixel(text, vec![input("TEXCOORD", 2, 0x3)], vec![target()], Vec::new());
    let shader = ProgramShader::from_ir(program.clone()).unwrap();
    let inputs = [bits([0.25, 0.75, 0.0, 0.0])];
    let e = shader.execute(&inputs, &[], &mut Unused).unwrap();
    assert_eq!(lanes(e.outputs[0]), [0.25, 0.75, 0.75, 0.25]);
    agree(&program, &e, &reference(&program, &inputs, &[], &mut no_samples));
}

/// The resources of [`array_program`]: four textures, and an unbounded array of samplers.
fn arrays(textures: u32) -> Vec<Resource> {
    vec![
        resource(
            0,
            "tex",
            ResourceKind::Texture {
                dim: TextureDim::D2,
                arrayed: false,
                multisampled: false,
                sample: SampleType::Float,
            },
            D3dClass::ShaderResource,
            0,
            textures,
        ),
        resource(
            1,
            "smp",
            ResourceKind::Sampler { comparison: false },
            D3dClass::Sampler,
            0,
            Resource::UNBOUNDED,
        ),
    ]
}

/// `SV_Target0 = tex[uint(x)].SampleLevel(smp[uint(x) + 1], (0.5, 0.5), 0)` with `textures` textures and an unbounded
/// sampler array: both picked by an index the program computes.
fn array_program(textures: u32) -> Program {
    let text = "iridia-ir 1
fn main {
  %0: f32 = input 0.x
  %1: u32 = convert.saturate %0
  %2: u32 = u32(1)
  %3: u32 = iadd %1, %2
  %4: vec2<f32> = vec2<f32>(0.5, 0.5)
  %5: vec4<f32> = sample r0@%1, r1@%3, %4 level 0
  output 0.xyzw = %5
  return
}
";
    pixel(text, vec![input("IDX", 1, 0x1)], vec![target()], arrays(textures))
}

/// Answers a sample with the elements it names: (texture element, sampler element, 0, 1).
struct Elements;
impl iridia_exec::Backend for Elements {
    fn sample(&mut self, r: ImageSample<iridia_exec::Element>) -> Result<[f32; 4], String> {
        assert_eq!(
            (r.texture.id, r.sampler.id),
            (
                iridia_ir::interface::ResourceId(0),
                iridia_ir::interface::ResourceId(1)
            )
        );
        Ok([r.texture.index as f32, r.sampler.index as f32, 0.0, 1.0])
    }
}

/// Four lanes, each with its own index: a divergent index picks a different element in each, and an index past the
/// array's four textures stops the run naming the resource and the index. The unbounded sampler array has no length
/// to check here (the caller's to check): sampler 5 is asked for by texture 4.
#[test]
fn an_index_the_program_computes_picks_an_element_of_a_resource_array() {
    use iridia_exec::{Executable, Resources, Runner};
    let program = array_program(4);
    let exe = Executable::from_ir(&program).unwrap();
    let resources = Resources::default();
    let bound = exe.bind_resources(&resources).unwrap();
    let mut backend = Elements;
    let mut quad = Runner::<4>::new(&exe, &bound, &mut backend, None);
    let arg = |x: f32| [x.to_bits(), 0, 0, 0];
    let out = quad
        .run([&[arg(0.0)][..], &[arg(3.0)][..], &[arg(1.0)][..], &[arg(2.0)][..]])
        .unwrap();
    let lanes: Vec<[f32; 4]> = out.iter().map(|o| lanes(o.values[0])).collect();
    assert_eq!(
        lanes,
        [
            [0.0, 1.0, 0.0, 1.0],
            [3.0, 4.0, 0.0, 1.0],
            [1.0, 2.0, 0.0, 1.0],
            [2.0, 3.0, 0.0, 1.0]
        ]
    );
    let mut one = Runner::<1>::new(&exe, &bound, &mut backend, None);
    let error = one.run([&[arg(4.0)][..]]).unwrap_err();
    assert!(
        error.contains("resource 0: element 4 of an array of 4"),
        "{error}"
    );
}

/// The interface's count is a resource array's length, the one place the IR states it, and a run checks an index
/// against it. The naga-based writer wrote the arrays as WGSL (`binding_array<texture_2d<f32>, 4>`,
/// `binding_array<sampler>`); the IR's lowering to naga takes no element of an array of resources yet, and says so.
#[test]
fn a_resource_array_is_written_and_its_count_is_checked() {
    let e = iridia_naga::wgsl(&array_program(4), &iridia_naga::Profile::NAGA).unwrap_err();
    assert!(
        e.to_string()
            .contains("a sample through an element of an array of samplers is not lowered yet"),
        "{e}"
    );
    // three textures: element 3 is past the array
    use iridia_exec::{Executable, Resources, Runner};
    let three = array_program(3);
    let exe = Executable::from_ir(&three).unwrap();
    let resources = Resources::default();
    let bound = exe.bind_resources(&resources).unwrap();
    let mut backend = Elements;
    let error = Runner::<1>::new(&exe, &bound, &mut backend, None)
        .run([&[[3f32.to_bits(), 0, 0, 0]][..]])
        .unwrap_err();
    assert!(
        error.contains("resource 0: element 3 of an array of 3"),
        "{error}"
    );
}

/// A pixel program over `uv` (TEXCOORD0, two lanes, `%0`) whose `SV_Target0` is the text's lines' last value,
/// `%result` in them.
fn program_of(body: &str, result: u32) -> Program {
    let text = format!(
        "iridia-ir 1
fn main {{
  %0: vec2<f32> = input 0.xy
  %1: f32 = %0.x
  %2: f32 = %0.y
{body}  output 0.xyzw = %{result}
  return
}}
"
    );
    pixel(&text, vec![input("TEXCOORD", 2, 0x3)], vec![target()], Vec::new())
}

/// The program's `SV_Target0` for `uv = (x, y)` in `iridia-exec`, which `iridia-exec-ir` agrees with.
fn run(program: Program, x: f32, y: f32) -> [f32; 4] {
    let shader = ProgramShader::from_ir(program.clone()).unwrap();
    let inputs = [bits([x, y, 0.0, 0.0])];
    let e = shader.execute(&inputs, &[], &mut Unused).unwrap();
    agree(&program, &e, &reference(&program, &inputs, &[], &mut no_samples));
    lanes(e.outputs[0])
}

/// A value loaded from a variable is what the variable held at the load: a store to the variable after the load does
/// not reach a lane or swizzle of the value read after the store (WGSL's `let s = l; l = ...; s.zw`), whether the read
/// follows the store directly or after a construct.
#[test]
fn a_loaded_value_keeps_what_the_local_held_at_the_load() {
    for after_a_construct in [false, true] {
        for by_swizzle in [false, true] {
            let construct = if after_a_construct {
                "  block '0 {\n    yield\n  }\n"
            } else {
                ""
            };
            let (read, result) = if by_swizzle {
                ("  %7: vec4<f32> = %5.wzyx\n".to_owned(), 7)
            } else {
                (
                    "  %7: f32 = %5.x\n  %8: f32 = %5.y\n  %9: f32 = %5.z\n  %10: f32 = %5.w\n  \
                     %11: vec4<f32> = compose %7, %8, %9, %10\n"
                        .to_owned(),
                    11,
                )
            };
            let text = format!(
                "iridia-ir 1
fn main {{
  var l: vec4<f32>
  %0: vec2<f32> = input 0.xy
  %1: f32 = %0.x
  %2: f32 = %0.y
  %3: f32 = f32(1.0)
  %4: vec4<f32> = compose %1, %2, %3, %2
  l = %4
  %5: vec4<f32> = load l
  %6: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0)
  l = %6
{construct}{read}  output 0.xyzw = %{result}
  return
}}
"
            );
            let program = pixel(&text, vec![input("TEXCOORD", 2, 0x3)], vec![target()], Vec::new());
            let want = if by_swizzle {
                [0.5, 1.0, 0.5, 0.25]
            } else {
                [0.25, 0.5, 1.0, 0.5]
            };
            assert_eq!(
                run(program, 0.25, 0.5),
                want,
                "after a construct {after_a_construct}, by swizzle {by_swizzle}"
            );
        }
    }
}

/// The inverse trigonometric and hyperbolic functions, the fused multiply-add and the float remainder, against the
/// values f64 gives rounded once (and C's fmod: the sign of the dividend).
#[test]
fn trigonometry_fma_and_the_float_remainder() {
    let unary = program_of(
        "  %3: f32 = facos %1
  %4: f32 = fasin %1
  %5: f32 = fatan %1
  %6: f32 = ftan %1
  %7: vec4<f32> = compose %3, %4, %5, %6
",
        7,
    );
    let hyperbolic = program_of(
        "  %3: f32 = fsinh %1
  %4: f32 = fcosh %1
  %5: f32 = ftanh %1
  %6: f32 = fsinh %1
  %7: vec4<f32> = compose %3, %4, %5, %6
",
        7,
    );
    let x = 0.3f32;
    let want = |f: fn(f64) -> f64| f(x as f64) as f32;
    assert_eq!(
        run(unary, x, 0.0),
        [want(f64::acos), want(f64::asin), want(f64::atan), want(f64::tan)]
    );
    assert_eq!(
        run(hyperbolic, x, 0.0),
        [want(f64::sinh), want(f64::cosh), want(f64::tanh), want(f64::sinh)]
    );
    let remainder = program_of(
        "  %3: f32 = frem %1, %2
  %4: f32 = fneg %1
  %5: f32 = frem %4, %2
  %6: f32 = f32(0.5)
  %7: f32 = fma %1, %2, %6
  %8: vec4<f32> = compose %3, %5, %7, %6
",
        8,
    );
    assert_eq!(
        run(remainder, 5.5, 2.0),
        [1.5, -1.5, 5.5f32.mul_add(2.0, 0.5), 0.5]
    );
}

/// A mix of vectors by a vector and by a scalar and a normalization, as a lifter writes them out (the IR has no `mix`
/// or `normalize`: HLSL's `lerp(v, w, a)` is `v + (w - v) * a`, `normalize(v)` is `v / sqrt(dot(v, v))`), `sign` and
/// the relational operations, against the values worked out here: with x = 0.5 and y = 2, v = (x, y, 2, 0.5) and
/// w = (y, x, -1, 4).
#[test]
fn mix_normalize_sign_and_the_relational_functions() {
    // v (%7) and w (%8)
    let vectors = "  %3: f32 = f32(2.0)
  %4: f32 = f32(0.5)
  %5: f32 = f32(-1.0)
  %6: f32 = f32(4.0)
  %7: vec4<f32> = compose %1, %2, %3, %4
  %8: vec4<f32> = compose %2, %1, %5, %6
";
    let mix = |a: &str| {
        program_of(
            &format!(
                "{vectors}{a}  %10: vec4<f32> = fsub %8, %7
  %11: vec4<f32> = fmul.ieee %10, %9
  %12: vec4<f32> = fadd %7, %11
"
            ),
            12,
        )
    };
    // by a scalar
    let by_scalar = mix("  %9: vec4<f32> = splat %1\n");
    assert_eq!(run(by_scalar, 0.5, 2.0), [1.25, 1.25, 0.5, 2.25]);
    // by a vector (x, x, y, y)
    let by_vector = mix("  %9: vec4<f32> = compose %1, %1, %2, %2\n");
    assert_eq!(run(by_vector, 0.5, 2.0), [1.25, 1.25, -4.0, 7.5]);
    let normalized = program_of(
        &format!(
            "{vectors}  %9: f32 = dot %7, %7
  %10: f32 = fsqrt %9
  %11: vec4<f32> = splat %10
  %12: vec4<f32> = fdiv %7, %11
"
        ),
        12,
    );
    let length = 8.5f64.sqrt() as f32;
    assert_eq!(
        run(normalized, 0.5, 2.0),
        [0.5 / length, 2.0 / length, 2.0 / length, 0.5 / length]
    );
    let signs = program_of(
        "  %3: f32 = f32(0.0)
  %4: f32 = fneg %1
  %5: f32 = fsign %1
  %6: f32 = fsign %4
  %7: f32 = fsign %3
  %8: f32 = fsign %2
  %9: vec4<f32> = compose %5, %6, %7, %8
",
        9,
    );
    assert_eq!(run(signs, 0.5, 2.0), [1.0, -1.0, 0.0, 1.0]);
    // any and all of v < w = (true, false, false, true); NaN from 0 / 0, infinity from x / 0
    let relational = program_of(
        &format!(
            "{vectors}  %9: vec4<bool> = fcmp.lt %7, %8
  %10: bool = any %9
  %11: bool = all %9
  %12: f32 = f32(0.0)
  %13: f32 = fdiv %12, %12
  %14: f32 = fdiv %1, %12
  %15: bool = isnan %13
  %16: bool = isinf %14
  %17: f32 = f32(1.0)
  %18: f32 = select %10, %17, %12
  %19: f32 = select %11, %17, %12
  %20: f32 = select %15, %17, %12
  %21: f32 = select %16, %17, %12
  %22: vec4<f32> = compose %18, %19, %20, %21
"
        ),
        22,
    );
    assert_eq!(run(relational, 0.5, 2.0), [1.0, 0.0, 1.0, 1.0]);
}

/// `pow` (in double precision, rounded once) and a boolean made a float and an integer, as the IR writes it (a select
/// of one or zero: the IR converts numbers only).
#[test]
fn pow_and_the_conversion_of_booleans() {
    let program = program_of(
        "  %3: f32 = fpow %2, %1
  %4: bool = fcmp.lt %1, %2
  %5: bool = fcmp.lt %2, %1
  %6: f32 = f32(1.0)
  %7: f32 = f32(0.0)
  %8: f32 = select %4, %6, %7
  %9: f32 = select %5, %6, %7
  %10: u32 = u32(1)
  %11: u32 = u32(0)
  %12: u32 = select %4, %10, %11
  %13: f32 = convert.saturate %12
  %14: vec4<f32> = compose %3, %8, %9, %13
",
        14,
    );
    assert_eq!(run(program, 0.5, 2.0), [2.0f64.powf(0.5) as f32, 1.0, 0.0, 1.0]);
}
