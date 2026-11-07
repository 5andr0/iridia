//! Compute programs lowered and run: each program in IR text is compiled into `iridia-exec` as the IR itself
//! (`Executable::from_ir`) and, written as WGSL by this crate and read back by naga's front end, as the naga module it
//! is; dispatched on the same memory, the two must leave the same memory. A property test generates the programs:
//! stores, loads and atomics of a raw buffer and of workgroup memory at addresses inside and past their ends, with
//! barriers between, each invocation storing what its loads and atomics gave at a word of its own.

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

/// The thread group: eight invocations along x; two groups are dispatched.
const GROUP: u32 = 8;
const GROUPS: [u32; 3] = [2, 1, 1];
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

/// The WGSL of `p` read back by naga's front end, as a program `iridia-exec` runs: the IR's interface, the entry point's
/// arguments named and bound as the interface gives them, and the buffer bound to the global its plan names.
fn from_wgsl(p: &Program, wgsl: &str) -> iridia::Program {
    let mut module = naga_fork::front::wgsl::parse_str(wgsl).unwrap_or_else(|e| panic!("{e}\n{wgsl}"));
    let io = iridia::entry_io(&p.interface).unwrap();
    let ep = module.entry_points.first_mut().unwrap();
    for (a, item) in ep.function.arguments.iter_mut().zip(&io.arguments) {
        a.name = Some(item.name.clone());
        a.binding = Some(item.binding.clone());
    }
    let buffer = module
        .global_variables
        .iter()
        .find(|(_, g)| g.name.as_deref().is_some_and(|n| n.trim_end_matches('_') == "u0"))
        .map(|(h, _)| h)
        .unwrap_or_else(|| panic!("no global u0\n{wgsl}"));
    iridia::Program {
        module,
        stage: Stage::Compute,
        entry: "main".into(),
        interface: p.interface.clone(),
        bindings: vec![(ResourceId(0), buffer)],
        conventions: iridia::Conventions::D3D,
        modes: iridia::D3dModes::default(),
        provenance: iridia::Provenance {
            format: ShaderFormat::Dxbc,
            target: "fixture".into(),
            lifter: "iridia-naga through WGSL".into(),
            lifter_version: 0,
            source_hash: None,
            instruction_count: None,
        },
        notes: Vec::new(),
    }
}

/// Textures are no compute program's here.
struct NoTextures;
impl iridia_exec::Backend for NoTextures {
    fn sample(&mut self, _: ImageSample<iridia_exec::Element>) -> Result<[f32; 4], String> {
        Err("no textures".into())
    }
}

/// The buffer as a dispatch of `exe` leaves it, from `words`.
fn dispatch(exe: &Executable, words: &[u32]) -> Result<Vec<u32>, String> {
    let resources = Resources::default();
    let bound = exe.bind_resources(&resources)?;
    let mut buffers = BTreeMap::from([(ResourceId(0), words.to_vec())]);
    let mut elements = BTreeMap::new();
    exe.dispatch_elements(&bound, &mut NoTextures, &mut buffers, &mut elements, GROUPS, 32)?;
    Ok(buffers.remove(&ResourceId(0)).unwrap_or_default())
}

/// The memory the IR leaves and the memory its WGSL leaves, from the same words; the WGSL with them.
fn both(p: &Program, words: &[u32]) -> (Vec<u32>, Vec<u32>, String) {
    let wgsl = iridia_naga::wgsl(p, &iridia_naga::Profile::NAGA).unwrap_or_else(|e| panic!("{e}"));
    let ir = Executable::from_ir(p).unwrap_or_else(|e| panic!("{e}"));
    let lowered = Executable::new(&from_wgsl(p, &wgsl)).unwrap_or_else(|e| panic!("{e}\n{wgsl}"));
    let a = dispatch(&ir, words).unwrap();
    let b = dispatch(&lowered, words).unwrap_or_else(|e| panic!("{e}\n{wgsl}"));
    (a, b, wgsl)
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

fn place() -> impl Strategy<Value = Place> {
    prop_oneof![
        (
            any::<bool>(),
            prop::sample::select(vec![0u32, 4, 8, 12]),
            prop::sample::select(vec![0u32, 4, 64, 1016, 1020, 1024, 4096])
        )
            .prop_map(|(local, stride, offset)| Place::Buffer {
                local,
                stride,
                offset
            }),
        (
            any::<bool>(),
            0..3u32,
            prop::sample::select(vec![0u32, 1, 7, 15, 16, 40])
        )
            .prop_map(|(local, scale, base)| Place::Group { local, scale, base }),
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
        let (ir, wgsl, written) = both(&p, &seeded_words(seed));
        prop_assert_eq!(ir, wgsl, "{}\n{}", text, written);
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
    let (ir, wgsl, written) = both(&p, &seeded_words(1));
    assert_eq!(ir, wgsl, "{written}");
    // every invocation stored 0 + 0 at its word
    assert!(ir.iter().take(16).all(|&w| w == 0), "{ir:?}");
}
