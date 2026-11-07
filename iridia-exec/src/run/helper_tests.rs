#![allow(clippy::unwrap_used)]
use super::*;
use iridia_ir::interface::{
    Access, BufferKind, Interface, Resource, ResourceKind, ResourceProvenance, SampleType, ScalarType,
    ShaderFormat, Stage, StageFacts, StorageTarget, TextureDim, Value,
};

#[derive(Default)]
struct Effects {
    stores: [usize; 3],
    atomics: [usize; 3],
    images: usize,
    image_atomics: usize,
    increments: usize,
    decrements: usize,
}
impl Backend for Effects {
    fn sample(&mut self, _: ImageSample<Element>) -> Result<[f32; 4], String> {
        panic!("no samples in this program")
    }
    fn buffer_words(&mut self, _: ResourceId) -> Result<u32, String> {
        Ok(16)
    }
    fn store_buffer(&mut self, id: ResourceId, _: u32, value: Reg, _: u8) -> Result<(), String> {
        assert_eq!(value[0], 1, "helper-private local stores still evaluate");
        self.stores[id.0 as usize] += 1;
        Ok(())
    }
    fn atomic_buffer(&mut self, request: BufferAtomic) -> Result<u32, String> {
        assert_eq!(request.value, 1);
        self.atomics[request.view.id.0 as usize] += 1;
        Ok(42)
    }
    fn store_image(&mut self, request: ImageStore<ResourceId>) -> Result<(), String> {
        assert_eq!(request.value[0], 1);
        self.images += 1;
        Ok(())
    }
    fn atomic_image(&mut self, request: ImageAtomic<ResourceId>) -> Result<u32, String> {
        assert_eq!(request.value, 1);
        self.image_atomics += 1;
        Ok(42)
    }
    fn counter(&mut self, _: Element, decrement: bool) -> Result<u32, String> {
        if decrement {
            self.decrements += 1;
        } else {
            self.increments += 1;
        }
        Ok(42)
    }
}

#[test]
fn helper_stores_atomics_and_both_counter_operations_have_no_effect() {
    let entry = iridia_ir::text::parse(
        "iridia-ir 1
fn main {
  var private: u32
  %0: u32 = u32(1)
  private = %0
  %1: u32 = load private
  %2: vec2<i32> = vec2<i32>(0, 0)
  %3: vec4<u32> = compose %1, %1, %1, %1
  store buffer r0 + 0 = %1
  store buffer r1 + 0 = %1
  store texel r2, %0 = %3
  store texel r3, %2 = %3
  %4: u32 = atomic.add buffer r0 + 0 value %1
  %5: u32 = atomic.add buffer r1 + 0 value %1
  %6: u32 = atomic.add buffer r2 + 0 value %1
  %7: u32 = atomic.add image r3, %2 value %1
  %8: u32 = counter.increment r1
  %9: u32 = counter.decrement r1
  return
}
",
    )
    .unwrap();
    let targets = [
        StorageTarget::Buffer(BufferKind::Raw),
        StorageTarget::Buffer(BufferKind::Structured { stride: 4 }),
        StorageTarget::Buffer(BufferKind::Typed(Value {
            scalar: ScalarType::Uint,
            width: 1,
        })),
        StorageTarget::Image {
            dim: TextureDim::D2,
            arrayed: false,
            sample: SampleType::Uint,
        },
    ];
    let program = iridia_ir::Program {
        stage: Stage::Pixel,
        entry,
        patch: None,
        workgroup: None,
        interface: Interface {
            stage: Stage::Pixel,
            inputs: vec![],
            outputs: vec![],
            resources: targets
                .into_iter()
                .enumerate()
                .map(|(i, target)| Resource {
                    id: ResourceId(i as u32),
                    kind: ResourceKind::Storage {
                        target,
                        access: Access::ReadWrite,
                    },
                    count: 1,
                    provenance: ResourceProvenance::default(),
                })
                .collect(),
            facts: StageFacts::None,
        },
        conventions: iridia_ir::Conventions::D3D,
        float: iridia_ir::FloatEnv::default(),
        provenance: iridia_ir::Provenance {
            format: ShaderFormat::Dxbc,
            target: "ps_5_0".into(),
            lifter: "test".into(),
            lifter_version: 0,
            source_hash: None,
            instruction_count: None,
            policies: vec![],
        },
        notes: vec![],
    };
    iridia_ir::validate(&program).unwrap();
    let executable = Executable::from_ir(&program).unwrap();
    let bound = executable.bind(|_| None, |_| None).unwrap();
    let mut effects = Effects::default();
    let mut runner = Runner::<4>::new(&executable, &bound, &mut effects, None);
    // Reuse the same runner: helper masks must be initialized afresh on every invocation.
    for helpers in [[false, true, false, true], [false; 4], [true; 4]] {
        let outputs = runner.run_masked([&[]; 4], 4, helpers).unwrap();
        assert_eq!(outputs.map(|o| o.discarded), helpers);
    }
    drop(runner);
    assert_eq!(effects.stores, [6; 3]);
    assert_eq!(effects.atomics, [6; 3]);
    assert_eq!(effects.images, 6);
    assert_eq!(effects.image_atomics, 6);
    assert_eq!(effects.increments, 6);
    assert_eq!(effects.decrements, 6);
}
