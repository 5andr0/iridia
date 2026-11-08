//! Compute programs lowered and run: each program in IR text is compiled into `iridia-exec` as the IR itself
//! (`Executable::from_ir`) and, written as WGSL by both writers, run on Direct3D 12's WARP through `tools.wgsl_run`
//! (where it is configured); dispatched on the same memory, the two must leave the same memory. Upstream naga reads and
//! validates this crate's text either way. A property test generates the programs: stores, loads and atomics of a raw
//! buffer and of workgroup memory at addresses inside and past their ends, with barriers between, each invocation
//! storing what its loads and atomics gave at a word of its own. A GPU runs invocations in no order the IR executor
//! shares, so every address a program touches inside its memory is one invocation's own (see [`place`]).

#![allow(clippy::unwrap_used, reason = "fixtures and generated programs")]

use std::collections::BTreeMap;
use std::fmt::Write as _;

use iridia_exec::{Executable, ImageSample, Resources};
use iridia_ir::{
    FloatEnv, Program, Provenance,
    interface::{
        Access, BufferKind, D3dBinding, D3dClass, Interface, Label, Resource, ResourceId, ResourceKind,
        ResourceProvenance, ScalarType, ShaderFormat, Stage, StageFacts, StorageTarget, SystemValue, Value,
        Varying, VaryingProvenance,
    },
};
use proptest::prelude::*;

mod inputs;
#[path = "support/warp.rs"]
mod warp;

/// The thread group: eight invocations along x; the generated programs dispatch one group.
const GROUP: u32 = 8;
const GROUPS: [u32; 3] = [1, 1, 1];
/// The words of the raw buffer `r0`: past them a read gives zero and a write does nothing.
const WORDS: u32 = 256;

fn id_input(name: &str, system: SystemValue) -> Varying {
    Varying {
        label: Label {
            name: name.into(),
            index: 0,
        },
        system: Some(system),
        value: Value {
            scalar: ScalarType::Uint,
            width: 3,
        },
        interp: None,
        used: true,
        provenance: VaryingProvenance::default(),
    }
}

/// A compute program of the text's function: the dispatch-wide and group-local invocation ids as inputs 0 and 1, and a
/// raw read-write buffer at `u0`.
fn program(text: &str) -> Program {
    let entry = iridia_ir::text::parse(text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    let p = Program {
        stage: Stage::Compute,
        entry,
        patch: None,
        workgroup: Some([GROUP, 1, 1]),
        interface: Interface {
            stage: Stage::Compute,
            inputs: vec![
                id_input("SV_DispatchThreadID", SystemValue::DispatchThreadId),
                id_input("SV_GroupThreadID", SystemValue::GroupThreadId),
            ],
            outputs: Vec::new(),
            resources: vec![Resource {
                id: ResourceId(0),
                kind: ResourceKind::Storage {
                    target: StorageTarget::Buffer(BufferKind::Raw),
                    access: Access::ReadWrite,
                },
                count: 1,
                provenance: ResourceProvenance {
                    name: "words".into(),
                    d3d: Some(D3dBinding {
                        class: D3dClass::UnorderedAccess,
                        register: 0,
                        space: 0,
                    }),
                    counter_of: None,
                    layout: None,
                },
            }],
            facts: StageFacts::default(),
        },
        conventions: iridia_ir::Conventions::D3D,
        float: FloatEnv::default(),
        provenance: Provenance {
            format: ShaderFormat::Dxbc,
            target: "fixture".into(),
            lifter: "text".into(),
            lifter_version: 1,
            source_hash: None,
            instruction_count: None,
            policies: Vec::new(),
        },
        notes: Vec::new(),
    };
    iridia_ir::validate(&p).unwrap_or_else(|e| panic!("{e:?}\n{text}"));
    p
}

/// The group and binding the WGSL gives the raw buffer `r0` (`u0` in the text), as the harness's resources name them.
fn buffer_slot(wgsl: &str) -> (u32, u32) {
    let at = wgsl
        .find("var<storage, read_write> u0")
        .unwrap_or_else(|| panic!("no u0\n{wgsl}"));
    // the attributes are on the declaration's line or the one before it
    let head = &wgsl[..at];
    let from = head
        .rfind("@group(")
        .unwrap_or_else(|| panic!("no @group( before u0\n{wgsl}"));
    let line = &head[from..];
    let number = |after: &str| {
        let rest = line
            .split(after)
            .nth(1)
            .unwrap_or_else(|| panic!("no {after} before u0\n{wgsl}"));
        rest.chars()
            .take_while(char::is_ascii_digit)
            .collect::<String>()
            .parse()
            .unwrap_or_else(|_| panic!("no number after {after}\n{wgsl}"))
    };
    (number("@group("), number("@binding("))
}

/// The buffer as the IR executor leaves it after dispatching `groups`, from `words`.
fn ir_memory(p: &Program, words: &[u32], groups: [u32; 3]) -> Vec<u32> {
    let exe = Executable::from_ir(p).unwrap_or_else(|e| panic!("{e}"));
    let resources = Resources::default();
    let bound = exe.bind_resources(&resources).unwrap();
    let mut buffers = BTreeMap::from([(ResourceId(0), words.to_vec())]);
    let mut elements = BTreeMap::new();
    exe.dispatch_elements(&bound, &mut NoTextures, &mut buffers, &mut elements, groups, 32)
        .unwrap();
    buffers.remove(&ResourceId(0)).unwrap_or_default()
}

/// Textures are no compute program's here.
struct NoTextures;
impl iridia_exec::Backend for NoTextures {
    fn sample(&mut self, _: ImageSample<iridia_exec::Element>) -> Result<[f32; 4], String> {
        Err("no textures".into())
    }
}

/// The buffer as WARP leaves it after running the WGSL `wgsl` of `p` over `groups`, from `words`.
fn warp_memory(warp: &mut warp::Warp, wgsl: &str, words: &[u32], groups: [u32; 3]) -> Vec<u32> {
    let (group, binding) = buffer_slot(wgsl);
    let reply = warp
        .run(&serde_json::json!({
            "id": 0,
            "wgsl": wgsl,
            "dispatch": groups,
            "resources": [{"group": group, "binding": binding, "buffer": {"words": words}}],
        }))
        .unwrap_or_else(|e| panic!("{e}\n{wgsl}"));
    reply["resources"][0]["words"]
        .as_array()
        .unwrap_or_else(|| panic!("no buffer in the reply {reply}\n{wgsl}"))
        .iter()
        .map(|w| u32::try_from(w.as_u64().unwrap()).unwrap())
        .collect()
}

/// Both writers' WGSL for a browser, read and validated by upstream naga (the profile's own checks for this crate's).
fn texts(p: &Program) -> [String; 2] {
    let ours = iridia_naga::wgsl(p, &iridia_naga::Profile::BROWSER).unwrap_or_else(|e| panic!("{e}"));
    let module =
        naga::front::wgsl::parse_str(&ours).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&ours)));
    iridia_naga::validate(&module, &iridia_naga::Profile::BROWSER).unwrap_or_else(|e| panic!("{e}\n{ours}"));
    let theirs = iridia_wgsl::wgsl(p, &iridia_wgsl::Profile::BROWSER).unwrap_or_else(|e| panic!("{e}"));
    [ours, theirs]
}

fn seeded_words(seed: u64) -> Vec<u32> {
    (0..WORDS)
        .map(|w| {
            let mut z = seed ^ u64::from(w).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            ((z ^ (z >> 31)) as u32) % 997
        })
        .collect()
}

/// Where a step reaches: a byte address `offset + stride * id` of the buffer, or element `id * scale + base` of the
/// workgroup variable (of 16 elements), `id` the invocation's dispatch-wide id or its id in the group.
#[derive(Clone, Copy, Debug)]
enum Place {
    Buffer { local: bool, stride: u32, offset: u32 },
    Group { local: bool, scale: u32, base: u32 },
}

#[derive(Clone, Copy, Debug)]
enum Step {
    Store(Place, u32),
    Load(Place),
    Atomic(Place, iridia_ir::AtomicOp, u32),
    Barrier,
}

const ATOMICS: [iridia_ir::AtomicOp; 11] = [
    iridia_ir::AtomicOp::Add,
    iridia_ir::AtomicOp::Sub,
    iridia_ir::AtomicOp::And,
    iridia_ir::AtomicOp::Or,
    iridia_ir::AtomicOp::Xor,
    iridia_ir::AtomicOp::SMin,
    iridia_ir::AtomicOp::SMax,
    iridia_ir::AtomicOp::UMin,
    iridia_ir::AtomicOp::UMax,
    iridia_ir::AtomicOp::Exchange,
    iridia_ir::AtomicOp::CompareExchange,
];

/// The places a program reaches. Inside its memory an invocation reaches only addresses of its own, so no two
/// invocations race and the answer does not depend on an order: a buffer address is `48 * id + offset` for an offset
/// within the 48 bytes (the accumulators are at the end, past every invocation's), a workgroup element `2 * id + base`
/// for a base of 0 or 1. Past the memory every invocation may reach: a buffer address past 1024 bytes, the word that
/// an offset of 1000, 1016 or 1020 puts only invocation 0 inside the buffer at, and a workgroup base of 16 or more.
fn place() -> impl Strategy<Value = Place> {
    prop_oneof![
        (
            any::<bool>(),
            prop::sample::select(vec![0u32, 4, 8, 12, 20, 44, 1000, 1016, 1020, 1024, 4096])
        )
            .prop_map(|(local, offset)| Place::Buffer {
                local,
                stride: 48,
                offset
            }),
        (any::<bool>(), prop::sample::select(vec![0u32, 1, 16, 40])).prop_map(|(local, base)| Place::Group {
            local,
            scale: 2,
            base
        }),
    ]
}

fn step() -> impl Strategy<Value = Step> {
    prop_oneof![
        (place(), 0..5u32).prop_map(|(p, k)| Step::Store(p, k)),
        place().prop_map(Step::Load),
        (place(), prop::sample::select(ATOMICS.to_vec()), 0..5u32)
            .prop_map(|(p, op, k)| Step::Atomic(p, op, k)),
        Just(Step::Barrier),
    ]
}

/// IR text being written: the lines so far and the next value's number.
struct Text {
    out: String,
    next: u32,
}

impl Text {
    /// A new value's number.
    fn fresh(&mut self) -> u32 {
        self.next += 1;
        self.next - 1
    }

    /// A `u32` operand of invocation `id` (a value): `id * 3 + k`, or for `k` 4 a large word, whose sign bit is set.
    fn operand(&mut self, k: u32, id: u32) -> u32 {
        if k == 4 {
            let (big, v) = (self.fresh(), self.fresh());
            let _ = writeln!(
                self.out,
                "  %{big}: u32 = u32(4294967200)
  %{v}: u32 = xor %{big}, %{id}"
            );
            return v;
        }
        let (scaled, lit, v) = (self.fresh(), self.fresh(), self.fresh());
        let _ = writeln!(
            self.out,
            "  %{scaled}: u32 = imul %{id}, %5
  %{lit}: u32 = u32({k})
  %{v}: u32 = iadd %{scaled}, %{lit}"
        );
        v
    }

    /// What a place is in the text: a buffer's address, or an element of `g` (its index computed first).
    fn reach(&mut self, p: Place) -> String {
        match p {
            Place::Buffer {
                local,
                stride,
                offset,
            } => {
                format!("buffer r0[%{} * {stride}] + {offset}", if local { 3 } else { 1 })
            }
            Place::Group { local, scale, base } => {
                let id = if local { 3 } else { 1 };
                let (s, scaled, b, i) = (self.fresh(), self.fresh(), self.fresh(), self.fresh());
                let _ = writeln!(
                    self.out,
                    "  %{s}: u32 = u32({scale})
  %{scaled}: u32 = imul %{id}, %{s}
  %{b}: u32 = u32({base})
                       %{i}: u32 = iadd %{scaled}, %{b}"
                );
                format!("g[%{i}]")
            }
        }
    }
}

/// The program of `steps` in IR text: each stored or atomic operand `id * 3 + k` ([`Text::operand`]), each loaded and
/// atomic result folded into an accumulator stored at word `WORDS - 32 + id` (each invocation's own).
fn text(steps: &[Step]) -> String {
    let mut t = Text {
        out: String::from(
            "iridia-ir 1
fn main {
  var g: [u32; 16] workgroup
",
        ),
        next: 6,
    };
    t.out.push_str(
        "  %0: vec3<u32> = input 0.xyz
  %1: u32 = %0.x
  %2: vec3<u32> = input 1.xyz
  %3: u32 = %2.x
",
    );
    t.out.push_str(
        "  %4: u32 = u32(0)
  %5: u32 = u32(3)
",
    );
    let mut acc = 4u32;
    for s in steps {
        match *s {
            Step::Barrier => t.out.push_str(
                "  barrier execution workgroup storage.workgroup
",
            ),
            Step::Store(p, k) => {
                let v = t.operand(k, 1);
                let at = t.reach(p);
                let _ = match p {
                    Place::Buffer { .. } => writeln!(t.out, "  store {at} = %{v}"),
                    Place::Group { .. } => writeln!(t.out, "  {at} = %{v} oob.zero"),
                };
            }
            Step::Load(p) => {
                let at = t.reach(p);
                let (x, folded) = (t.fresh(), t.fresh());
                let _ = match p {
                    Place::Buffer { .. } => writeln!(t.out, "  %{x}: u32 = {at} oob.zero"),
                    Place::Group { .. } => writeln!(t.out, "  %{x}: u32 = load {at} oob.zero"),
                };
                let _ = writeln!(t.out, "  %{folded}: u32 = xor %{acc}, %{x}");
                acc = folded;
            }
            Step::Atomic(p, op, k) => {
                let v = t.operand(k, 1);
                let c = t.operand((k + 1) % 5, 3);
                let at = t.reach(p);
                let memory = match p {
                    Place::Buffer { .. } => at,
                    Place::Group { .. } => format!("var {at}"),
                };
                let (x, folded) = (t.fresh(), t.fresh());
                let name = format!("{op:?}").to_lowercase();
                let compare = if op == iridia_ir::AtomicOp::CompareExchange {
                    format!(" compare %{c}")
                } else {
                    String::new()
                };
                let _ = writeln!(t.out, "  %{x}: u32 = atomic.{name} {memory} value %{v}{compare}");
                let _ = writeln!(t.out, "  %{folded}: u32 = xor %{acc}, %{x}");
                acc = folded;
            }
        }
    }
    // each invocation's accumulator at a word of its own
    let _ = writeln!(t.out, "  store buffer r0[%1 * 4] + {} = %{acc}", (WORDS - 32) * 4);
    t.out.push_str(
        "  return
}
",
    );
    t.out
}

/// The WARP half of a case: for each writer's text, WARP leaves from `words` the memory the IR executor leaves.
fn agree(p: &Program, text: &str, words: &[u32], groups: [u32; 3], warp: Option<&mut warp::Warp>) {
    let ir = ir_memory(p, words, groups);
    let [ours, theirs] = texts(p);
    if let Some(warp) = warp {
        for (writer, wgsl) in [("iridia-naga", ours), ("iridia-wgsl", theirs)] {
            assert_eq!(
                ir,
                warp_memory(warp, &wgsl, words, groups),
                "{writer}\n{text}\n{wgsl}"
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Any program of buffer and workgroup memory steps leaves the same memory written as WGSL as the IR leaves: the
    /// bounds of each memory, each atomic operation and its result (a compare-exchange strong, a minimum or maximum of
    /// the other signedness than its words'), and workgroup memory starting as zero.
    #[test]
    fn lowered_memory_steps_leave_the_memory_the_ir_leaves(
        steps in prop::collection::vec(step(), 1..12),
        seed in any::<u64>(),
    ) {
        let text = text(&steps);
        let p = program(&text);
        let mut warp = warp::Warp::shared();
        agree(&p, &text, &seeded_words(seed), GROUPS, warp.as_deref_mut());
    }
}

/// A word stored past the buffer's end is not written and a read there gives zero; an atomic there changes nothing and
/// gives zero.
#[test]
fn memory_past_the_end_is_not_reached() {
    let text = "iridia-ir 1
fn main {
  %0: vec3<u32> = input 0.xyz
  %1: u32 = %0.x
  %2: vec3<u32> = input 1.xyz
  %3: u32 = u32(7)
  store buffer r0 + 1024 = %3
  %4: u32 = buffer r0 + 1024 oob.zero
  %5: u32 = atomic.add buffer r0 + 4096 value %3
  %6: u32 = iadd %4, %5
  store buffer r0[%1 * 4] + 0 = %6
  return
}
";
    let p = program(text);
    let mut warp = warp::Warp::shared();
    agree(&p, text, &seeded_words(1), GROUPS, warp.as_deref_mut());
    let ir = ir_memory(&p, &seeded_words(1), GROUPS);
    // every invocation stored 0 + 0 at its word
    assert!(ir.iter().take(8).all(|&w| w == 0), "{ir:?}");
}

/// Two groups: each invocation has its dispatch-wide id and its id in the group, and each group its own workgroup memory.
#[test]
fn groups_have_their_own_ids_and_workgroup_memory() {
    let text = "iridia-ir 1
fn main {
  var g: [u32; 16] workgroup
  %0: vec3<u32> = input 0.xyz
  %1: u32 = %0.x
  %2: vec3<u32> = input 1.xyz
  %3: u32 = %2.x
  g[%3] = %1 oob.zero
  barrier execution workgroup storage.workgroup
  %4: u32 = load g[%3] oob.zero
  store buffer r0[%1 * 4] + 0 = %4
  store buffer r0[%1 * 4] + 64 = %3
  return
}
";
    let p = program(text);
    let groups = [2, 1, 1];
    let words = seeded_words(2);
    let ir = ir_memory(&p, &words, groups);
    // the first 16 words are the dispatch-wide ids (each group's memory held its own invocations' ids), the next 16 the ids
    // in the group
    let want: Vec<u32> = (0..16).chain((0..16).map(|g| g % GROUP)).collect();
    assert_eq!(ir.get(..32), Some(want.as_slice()), "{ir:?}");
    let mut warp = warp::Warp::shared();
    agree(&p, text, &words, groups, warp.as_deref_mut());
}
