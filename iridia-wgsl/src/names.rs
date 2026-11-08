//! Names: every name the text declares from something a source called (a varying's label, a variable, a resource) is
//! made a WGSL identifier that is none of WGSL's keywords or reserved words, none of its predeclared names (a variable
//! called `min` would hide the function the text calls) and none of the names the writer itself makes (`v3` for a value,
//! `x7` for a temporary, `iridia_` for a helper function).

use alloc::{
    collections::BTreeSet,
    format,
    string::{String, ToString},
};

/// WGSL's keywords and reserved words (the specification's lists), sorted.
const RESERVED: &[&str] = &[
    "NULL",
    "Self",
    "abstract",
    "active",
    "alias",
    "alignas",
    "alignof",
    "as",
    "asm",
    "asm_fragment",
    "async",
    "attribute",
    "auto",
    "await",
    "become",
    "binding_array",
    "break",
    "case",
    "cast",
    "catch",
    "class",
    "co_await",
    "co_return",
    "co_yield",
    "coherent",
    "column_major",
    "common",
    "compile",
    "compile_fragment",
    "concept",
    "const",
    "const_assert",
    "const_cast",
    "consteval",
    "constexpr",
    "constinit",
    "continue",
    "continuing",
    "crate",
    "debugger",
    "decltype",
    "default",
    "delete",
    "demote",
    "demote_to_helper",
    "diagnostic",
    "discard",
    "do",
    "dynamic_cast",
    "else",
    "enable",
    "enum",
    "explicit",
    "export",
    "extends",
    "extern",
    "external",
    "fallthrough",
    "false",
    "filter",
    "final",
    "finally",
    "fn",
    "for",
    "friend",
    "from",
    "fxgroup",
    "get",
    "goto",
    "groupshared",
    "highp",
    "if",
    "impl",
    "implements",
    "import",
    "inline",
    "instanceof",
    "interface",
    "layout",
    "let",
    "loop",
    "lowp",
    "macro",
    "macro_rules",
    "match",
    "mediump",
    "meta",
    "mod",
    "module",
    "move",
    "mut",
    "mutable",
    "namespace",
    "new",
    "nil",
    "noexcept",
    "noinline",
    "nointerpolation",
    "noperspective",
    "null",
    "nullptr",
    "of",
    "operator",
    "override",
    "package",
    "packoffset",
    "partition",
    "pass",
    "patch",
    "pixelfragment",
    "precise",
    "precision",
    "premerge",
    "priv",
    "protected",
    "pub",
    "public",
    "readonly",
    "ref",
    "regardless",
    "register",
    "reinterpret_cast",
    "require",
    "requires",
    "resource",
    "restrict",
    "return",
    "self",
    "set",
    "shared",
    "sizeof",
    "smooth",
    "snorm",
    "static",
    "static_assert",
    "static_cast",
    "std",
    "struct",
    "subroutine",
    "super",
    "switch",
    "target",
    "template",
    "this",
    "thread_local",
    "throw",
    "trait",
    "true",
    "try",
    "type",
    "typedef",
    "typeid",
    "typename",
    "typeof",
    "union",
    "unless",
    "unorm",
    "unsafe",
    "unsized",
    "use",
    "using",
    "var",
    "varying",
    "virtual",
    "volatile",
    "wgsl",
    "where",
    "while",
    "with",
    "writeonly",
    "yield",
];

/// WGSL's predeclared functions, types and enumerants, sorted.
const PREDECLARED: &[&str] = &[
    "RayDesc",
    "RayIntersection",
    "abs",
    "acceleration_structure",
    "acos",
    "acosh",
    "all",
    "any",
    "array",
    "arrayLength",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "atomic",
    "bgra8unorm",
    "bitcast",
    "bool",
    "ceil",
    "clamp",
    "cos",
    "cosh",
    "countLeadingZeros",
    "countOneBits",
    "countTrailingZeros",
    "cross",
    "degrees",
    "determinant",
    "distance",
    "dot",
    "dpdx",
    "dpdxCoarse",
    "dpdxFine",
    "dpdy",
    "dpdyCoarse",
    "dpdyFine",
    "exp",
    "exp2",
    "extractBits",
    "f16",
    "f32",
    "faceForward",
    "firstLeadingBit",
    "firstTrailingBit",
    "floor",
    "fma",
    "fract",
    "frexp",
    "function",
    "fwidth",
    "fwidthCoarse",
    "fwidthFine",
    "getCandidateHitVertexPositions",
    "getCommittedHitVertexPositions",
    "i32",
    "insertBits",
    "inverseSqrt",
    "ldexp",
    "length",
    "log",
    "log2",
    "max",
    "min",
    "mix",
    "modf",
    "normalize",
    "private",
    "ptr",
    "quantizeToF16",
    "radians",
    "ray_query",
    "read",
    "read_write",
    "reflect",
    "refract",
    "reverseBits",
    "round",
    "sampler",
    "sampler_comparison",
    "saturate",
    "select",
    "sign",
    "sin",
    "sinh",
    "smoothstep",
    "sqrt",
    "step",
    "storage",
    "storageBarrier",
    "tan",
    "tanh",
    "transpose",
    "trunc",
    "u32",
    "uniform",
    "workgroup",
    "workgroupBarrier",
    "workgroupUniformLoad",
    "write",
];

/// Prefixes of names WGSL predeclares in families (`vec3f`, `mat4x4<f32>`, `textureSample`, `subgroupAdd`, `pack2x16float`,
/// `atomicAdd`, `quadSwapX`, `r32uint`) and of the names the writer makes itself.
const PREFIXES: &[&str] = &[
    "vec", "mat", "texture", "subgroup", "quad", "pack", "unpack", "atomic", "dot4", "rgba", "rgb", "rg",
    "r8", "r16", "r32", "bgra", "iridia_", "__", "rayQuery", "RAY_",
];

/// Whether `name` is of the shape of a name the writer makes: a value `v3`, a temporary `x7`, a parameter's storage `p2`.
fn made_by_the_writer(name: &str) -> bool {
    // `v3`, `x7`, `p2`, and a lane of one: `p2_1`, `q3_0`
    let mut chars = name.chars();
    let (digits, lane) = match chars.next() {
        Some('v' | 'x' | 'p' | 'o' | 'a' | 'q') => match chars.as_str().split_once('_') {
            Some((d, l)) => (d, Some(l)),
            None => (chars.as_str(), None),
        },
        _ => return false,
    };
    let all_digits = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit());
    all_digits(digits) && lane.is_none_or(all_digits)
}

/// Whether `name` may not be declared as it is.
fn taken(name: &str) -> bool {
    RESERVED.binary_search(&name).is_ok()
        || PREDECLARED.binary_search(&name).is_ok()
        || PREFIXES.iter().any(|p| name.starts_with(p))
        || made_by_the_writer(name)
}

/// A WGSL identifier made of `name`: its characters that WGSL allows, a letter first, and none of the above.
pub(crate) fn sanitise(name: &str) -> String {
    let mut out: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if !out.starts_with(|c: char| c.is_ascii_alphabetic()) {
        out.insert_str(0, "n_");
    }
    // a family's prefix cannot be escaped by a suffix (`vec2_` still starts with `vec`): move the name off it
    if PREFIXES.iter().any(|p| out.starts_with(p)) {
        out.insert_str(0, "n_");
    }
    while taken(&out) {
        out.push('_');
    }
    out
}

/// The names declared so far in one scope, each made different from the others.
#[derive(Debug, Default)]
pub(crate) struct Names {
    used: BTreeSet<String>,
}

impl Names {
    /// A name for `wanted`, different from every one given before.
    pub(crate) fn unique(&mut self, wanted: &str) -> String {
        let base = sanitise(wanted);
        let mut name = base.clone();
        let mut n = 1u32;
        while self.used.contains(&name) {
            name = format!("{base}_{n}");
            n = n.saturating_add(1);
        }
        self.used.insert(name.clone());
        name
    }

    /// Reserve a name the writer picked itself.
    pub(crate) fn reserve(&mut self, name: &str) {
        self.used.insert(name.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lists_are_sorted() {
        assert!(RESERVED.windows(2).all(|w| w[0] < w[1]));
        assert!(PREDECLARED.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn a_name_is_made_safe() {
        assert_eq!(sanitise("TEXCOORD0"), "TEXCOORD0");
        assert_eq!(sanitise("SV_Target0"), "SV_Target0");
        assert_eq!(sanitise("loop"), "loop_");
        assert_eq!(sanitise("v5"), "v5_");
        assert_eq!(sanitise("vec2"), "n_vec2");
        assert_eq!(sanitise("texture0"), "n_texture0");
        assert_eq!(sanitise("log2"), "log2_");
        assert_eq!(sanitise("0abc"), "n_0abc");
        assert_eq!(sanitise("a.b-c"), "a_b_c");
        assert_eq!(sanitise("_x"), "n__x");
    }

    #[test]
    fn two_equal_names_are_told_apart() {
        let mut n = Names::default();
        assert_eq!(n.unique("cb0"), "cb0");
        assert_eq!(n.unique("cb0"), "cb0_1");
        assert_eq!(n.unique("cb0"), "cb0_2");
    }
}
