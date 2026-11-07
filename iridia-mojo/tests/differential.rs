//! Random pixel programs of the dialect both lifts read (assignments through write masks, swizzles, the arithmetic,
//! the built-in functions, `?:`, `if`/`else`, `rep` loops, `discard`, samples), lifted by both: the IR lift, as it is,
//! after the passes and compiled into `iridia-exec`, runs as the older lift does on seeded inputs (`tests/pairing`).
//!
//! The IR lift must lift and run every program the grammar makes; the older lift is the reference where it has an
//! answer, and a program it refuses or `iridia-exec` cannot compile from it is set aside (it takes the replicated lane
//! of `cross(a.xxx, b)` for a float, for one). The grammar keeps clear of what the two may rightly compute otherwise: `pow` and `normalize` (the older lift's are
//! naga's, which `iridia-exec` computes otherwise: `normalize` as `x / sqrt(dot(x, x))`, where the IR's is Direct3D 9's
//! `nrm`, `x * inversesqrt(dot(x, x))`, the instruction MojoShader writes it for; `tests/ir.rs` and `tests/oracle.rs`
//! hold the IR's to the Direct3D 9 programs) and a NaN or infinity reaching `min`, `max` or `clamp` (the result is left open):
//! a function defined on part of the line reads `abs(x) + 1.0`, and a quotient's divisor too.
mod pairing;

use iridia::Stage;
use iridia_mojo::{Hints, Sampler};
use proptest::prelude::*;
use proptest::strategy::BoxedStrategy;

const HEADER: &str = "uniform vec4 ps_uniforms_vec4[4];
const vec4 ps_c4 = vec4(0.5, -1.0, 2.0, 0.25);
const ivec4 ps_i0 = ivec4(3, 2, 0, 0);
vec4 ps_r0;
vec4 ps_r1;
vec4 ps_r2;
#define ps_c0 ps_uniforms_vec4[0]
#define ps_c1 ps_uniforms_vec4[1]
#define ps_c2 ps_uniforms_vec4[2]
#define ps_c3 ps_uniforms_vec4[3]
uniform sampler2D ps_s0;
varying vec4 vTexCoord0;
#define ps_v0 vTexCoord0
varying vec4 vFrontColor;
#define ps_v1 vFrontColor
#define ps_oC0 gl_FragColor

void main()
{
\tps_r0 = ps_v0;
\tps_r1 = ps_v1;
\tps_r2 = ps_c0;
";

/// A swizzle of `width` lanes.
fn swizzle(width: usize) -> impl Strategy<Value = String> {
    proptest::collection::vec(0..4usize, width)
        .prop_map(|lanes| lanes.iter().map(|&l| ['x', 'y', 'z', 'w'][l]).collect())
}

fn literal() -> impl Strategy<Value = String> {
    prop::sample::select(&["0.0", "0.5", "1.0", "-1.0", "2.0", "0.25", "3.0", "-0.125"][..])
        .prop_map(str::to_owned)
}

/// The GLSL type of `width` floats.
fn ty(width: usize) -> String {
    if width == 1 {
        "float".into()
    } else {
        format!("vec{width}")
    }
}

/// A value of `width` floats read from the program's state or written out.
fn leaf(width: usize) -> BoxedStrategy<String> {
    let names = prop::sample::select(
        &[
            "ps_r0", "ps_r1", "ps_r2", "ps_v0", "ps_v1", "ps_c0", "ps_c1", "ps_c3", "ps_c4",
        ][..],
    );
    prop_oneof![
        3 => (names, swizzle(width)).prop_map(|(n, s)| format!("{n}.{s}")),
        1 => proptest::collection::vec(literal(), width).prop_map(move |l| {
            if width == 1 { l[0].clone() } else { format!("{}({})", ty(width), l.join(", ")) }
        }),
    ]
    .boxed()
}

/// A value of `width` floats, `depth` operations deep at most; with `sample`, it may sample the texture (a sample at an
/// implicit level is where every lane of the quad runs, as fxc compiles one: in a branch the lanes take apart its
/// derivatives are undefined).
fn expr(width: usize, depth: u32, sample: bool) -> BoxedStrategy<String> {
    if depth == 0 {
        return leaf(width);
    }
    let e = move |w: usize| expr(w, depth - 1, sample);
    let unary = prop::sample::select(&["abs", "fract", "floor", "ceil", "sign", "sin", "cos", "exp2"][..]);
    let partial = prop::sample::select(&["sqrt", "inversesqrt", "log2"][..]);
    let operator = prop::sample::select(&["+", "-", "*"][..]);
    let comparison = prop::sample::select(&["<", ">", "<=", ">=", "==", "!="][..]);
    let mut options: Vec<(u32, BoxedStrategy<String>)> = vec![
        (4, leaf(width)),
        (
            4,
            (e(width), operator, e(width))
                .prop_map(|(a, o, b)| format!("({a} {o} {b})"))
                .boxed(),
        ),
        (
            1,
            (e(width), e(width))
                .prop_map(|(a, b)| format!("({a} / (abs({b}) + 1.0))"))
                .boxed(),
        ),
        (
            2,
            (unary, e(width)).prop_map(|(f, a)| format!("{f}({a})")).boxed(),
        ),
        (
            1,
            (partial, e(width))
                .prop_map(|(f, a)| format!("{f}(abs({a}) + 1.0)"))
                .boxed(),
        ),
        (
            1,
            (prop::bool::ANY, e(width), e(width))
                .prop_map(|(min, a, b)| format!("{}({a}, {b})", if min { "min" } else { "max" }))
                .boxed(),
        ),
        (
            1,
            (e(width), e(1), e(1))
                .prop_map(|(x, a, b)| format!("clamp({x}, {a}, {b})"))
                .boxed(),
        ),
        (
            1,
            (e(width), e(width), e(1))
                .prop_map(|(x, y, a)| format!("mix({x}, {y}, {a})"))
                .boxed(),
        ),
        (
            1,
            (e(1), comparison, e(1), e(width), e(width))
                .prop_map(|(p, c, q, a, b)| format!("(({p} {c} {q}) ? {a} : {b})"))
                .boxed(),
        ),
    ];
    if sample {
        options.push((
            1,
            (e(2), swizzle(width))
                .prop_map(|(uv, s)| format!("texture2D(ps_s0, {uv}).{s}"))
                .boxed(),
        ));
    }
    if width == 1 {
        // `dp2`, `dp3` and `dp4` of vectors, which is all MojoShader writes `dot` for (the older lift's `dot` of two
        // floats is a `Dot` naga refuses)
        options.push((
            1,
            (2..=4usize)
                .prop_flat_map(move |w| (e(w), e(w)))
                .prop_map(|(a, b)| format!("dot({a}, {b})"))
                .boxed(),
        ));
        // no `length`: MojoShader never writes it (the older lift types it as its argument)
    } else {
        // a constructor of a lane and the rest
        let rest = width - 1;
        options.push((
            1,
            (e(1), e(rest))
                .prop_map(move |(a, b)| format!("{}({a}, {b})", ty(width)))
                .boxed(),
        ));
    }
    if width == 3 {
        options.push((
            1,
            (e(3), e(3)).prop_map(|(a, b)| format!("cross({a}, {b})")).boxed(),
        ));
    }
    prop::strategy::Union::new_weighted(options).boxed()
}

/// A statement, `depth` constructs deep at most; a loop's counter is named by its depth. Only a statement of `main`'s
/// own (`top`) samples.
fn statement(depth: u32, top: bool) -> BoxedStrategy<String> {
    let mask = (1..16u8).prop_map(|m| {
        (0..4)
            .filter(|l| m & (1 << l) != 0)
            .map(|l| ['x', 'y', 'z', 'w'][l])
            .collect::<String>()
    });
    let assignment = (0..3u8, mask)
        .prop_flat_map(move |(r, mask)| (Just(r), Just(mask.clone()), expr(mask.len(), 2, top)))
        .prop_map(|(r, mask, e)| {
            if mask == "xyzw" {
                format!("ps_r{r} = {e};")
            } else {
                format!("ps_r{r}.{mask} = {e};")
            }
        });
    if depth == 0 {
        return assignment.boxed();
    }
    let block =
        move || proptest::collection::vec(statement(depth - 1, false), 1..3).prop_map(|s| s.join("\n"));
    let condition = (
        expr(1, 1, top),
        prop::sample::select(&["<", ">=", "!="][..]),
        expr(1, 1, top),
    )
        .prop_map(|(a, c, b)| format!("{a} {c} {b}"));
    let mut options: Vec<(u32, BoxedStrategy<String>)> = vec![
        (4, assignment.boxed()),
        (
            1,
            (condition.clone(), block(), block())
                .prop_map(|(c, a, b)| {
                    format!(
                        "if ({c}) {{
{a}
}} else {{
{b}
}}"
                    )
                })
                .boxed(),
        ),
        (1, condition.prop_map(|c| format!("if ({c}) discard;")).boxed()),
    ];
    // a loop only in `main`'s own statements, where every lane of the quad iterates alike: `iridia-exec-ir` meets a
    // sample only of lanes that ran as many iterations, and refuses a quad that ran a loop in one branch of an `if`
    if top {
        options.push((
            1,
            (prop::sample::select(&["x", "y"][..]), block())
                .prop_map(move |(lane, b)| {
                    format!(
                        "for (int rep{depth} = 0; rep{depth} < ps_i0.{lane}; rep{depth}++) {{
{b}
}}"
                    )
                })
                .boxed(),
        ));
    }
    prop::strategy::Union::new_weighted(options).boxed()
}

fn program() -> impl Strategy<Value = String> {
    proptest::collection::vec(statement(2, true), 1..6)
        .prop_map(|body| format!("{HEADER}{}\n\tps_oC0 = ps_r0;\n}}\n", body.join("\n")))
}

fn hints() -> Hints {
    Hints {
        target: "test".into(),
        samplers: vec![Sampler {
            glsl_name: "ps_s0".into(),
            slot: 0,
            name: "Albedo".into(),
        }],
        ..Hints::default()
    }
}

proptest! {
    // a failing program is printed whole; none is saved into the source tree
    #![proptest_config(ProptestConfig { cases: 128, failure_persistence: None, ..ProptestConfig::default() })]

    #[test]
    fn the_ir_lift_runs_as_the_older_lift_does(source in program()) {
        match pairing::lifts_agree(&source, Stage::Pixel, &hints(), 4) {
            Ok(None) => {}
            // the older lift is the reference only where it has an answer; the IR lift must have one everywhere
            Ok(Some(why)) => prop_assume!(false, "{}", why),
            Err(e) => prop_assert!(false, "{e}\n{source}"),
        }
    }
}
