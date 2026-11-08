#![allow(clippy::unwrap_used)]
//! A memory fence imposes no execution rendezvous. The host already provides sequentially consistent memory.
mod support;

use std::collections::BTreeMap;

use iridia_exec::{Backend, Element, Executable, ImageSample, Resources};
use iridia_ir::Program;
use iridia_ir::interface::{
    Access, BufferKind, D3dClass, Interface, ResourceId, ResourceKind, ScalarType, ShaderFormat, Stage,
    StageFacts, StorageTarget, SystemValue,
};

fn program(barrier: &str) -> Program {
    let text = format!(
        "iridia-ir 1\nfn main {{\n  %0: u32 = input 0.x\n  %1: u32 = u32(0)\n  %2: bool = ucmp.eq %0, %1\n  if %2 {{\n    barrier {barrier}\n    yield\n  }} else {{\n    yield\n  }}\n  %3: u32 = input 1.x\n  store buffer r0[%3 * 4] + 0 = %3\n  return\n}}\n"
    );
    let interface = Interface {
        stage: Stage::Compute,
        inputs: vec![
            support::varying(
                "SV_GroupThreadID",
                0,
                ScalarType::Uint,
                3,
                None,
                Some(SystemValue::GroupThreadId),
            ),
            support::varying(
                "SV_DispatchThreadID",
                0,
                ScalarType::Uint,
                3,
                None,
                Some(SystemValue::DispatchThreadId),
            ),
        ],
        outputs: Vec::new(),
        resources: vec![support::resource(
            0,
            "words",
            ResourceKind::Storage {
                target: StorageTarget::Buffer(BufferKind::Raw),
                access: Access::ReadWrite,
            },
            D3dClass::UnorderedAccess,
            0,
            1,
        )],
        facts: StageFacts::default(),
    };
    let mut p = support::unchecked(support::function(&text), interface, ShaderFormat::Dxbc, "cs_5_0");
    p.workgroup = Some([8, 1, 1]);
    iridia_ir::validate(&p).unwrap();
    p
}

struct NoTextures;
impl Backend for NoTextures {
    fn sample(&mut self, _: ImageSample<Element>) -> Result<[f32; 4], String> {
        Err("no textures in this fixture".into())
    }
}

fn dispatch(p: &Program) -> Result<Vec<u32>, String> {
    let exe = Executable::from_ir(p).map_err(|e| e.to_string())?;
    let resources = Resources::default();
    let bound = exe.bind_resources(&resources)?;
    let mut buffers = BTreeMap::from([(ResourceId(0), vec![0x1234_5678; 32])]);
    exe.dispatch_elements(
        &bound,
        &mut NoTextures,
        &mut buffers,
        &mut BTreeMap::new(),
        [2, 1, 1],
        32,
    )?;
    Ok(buffers.remove(&ResourceId(0)).unwrap())
}

#[test]
fn a_memory_only_fence_may_be_reached_by_one_invocation() {
    let expected: Vec<u32> = (0..16).chain([0x1234_5678; 16]).collect();
    for fence in [
        "workgroup",
        "storage.workgroup",
        "storage.device",
        "workgroup storage.device",
    ] {
        let p = program(fence);
        let source = p.clone();
        assert_eq!(dispatch(&p).unwrap(), expected, "{fence}");
        assert_eq!(p, source, "execution must not rewrite the source barrier");
    }
}

#[test]
fn an_execution_barrier_still_requires_all_active_invocations() {
    let error = dispatch(&program("execution workgroup")).unwrap_err();
    assert!(
        error.contains("a barrier that not every invocation of the group reaches"),
        "{error}"
    );
}
