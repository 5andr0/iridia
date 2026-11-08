//! The IR lowered to stock naga (`iridia_naga`), written as WGSL and read back by naga's front end, as a program the
//! executor of naga-path programs runs with the older lift's interface: the IR gate's column C (`ir_gate.rs`) and the
//! lowered column of the SM 5.1 execution fixtures (`sm51_exec.rs`).
#![allow(dead_code)]

use naga as oracle_naga;
#[path = "inline.rs"]
mod pure_helpers;

/// What naga's WGSL front end makes of the lowering's text cannot say everything the older module says, which the
/// executor reads: put back from the older module by position, a global's name ending in a digit (naga's namer writes
/// `cb0_` for `cb0`) and the entry point's argument and result member names and bindings (WGSL cannot say a varying
/// has no interpolation, so the front end supplies the default). Anything else that differs is left for the executor
/// to refuse or to compute differently.
pub fn normalise(m: &mut naga::Module, old: &naga::Module) -> Result<(), String> {
    for (_, g) in m.global_variables.iter_mut() {
        if let Some(n) = &mut g.name
            && n.len() > 1
            && n.ends_with('_')
            && n[..n.len() - 1].ends_with(|c: char| c.is_ascii_digit())
        {
            n.pop();
        }
    }
    let (Some(ep), Some(oe)) = (m.entry_points.first_mut(), old.entry_points.first()) else {
        return Err("no entry point".into());
    };
    if ep.function.arguments.len() != oe.function.arguments.len() {
        return Err(format!(
            "{} arguments, where A's entry point has {}",
            ep.function.arguments.len(),
            oe.function.arguments.len()
        ));
    }
    for (a, o) in ep.function.arguments.iter_mut().zip(&oe.function.arguments) {
        a.name.clone_from(&o.name);
        a.binding.clone_from(&o.binding);
    }
    match (&ep.function.result, &oe.function.result) {
        (None, None) => Ok(()),
        (Some(r), Some(or)) => {
            let (rt, ort) = (r.ty, or.ty);
            let naga::TypeInner::Struct { members: om, .. } = &old.types[ort].inner else {
                return Err("A's result is not a structure".into());
            };
            let mut t = m.types[rt].clone();
            let naga::TypeInner::Struct { members, .. } = &mut t.inner else {
                return Err("the result is not a structure".into());
            };
            if members.len() != om.len() {
                return Err(format!(
                    "{} result members, where A's has {}",
                    members.len(),
                    om.len()
                ));
            }
            for (a, o) in members.iter_mut().zip(om) {
                a.name.clone_from(&o.name);
                a.binding.clone_from(&o.binding);
            }
            t.name.clone_from(&old.types[ort].name);
            m.types.replace(rt, t);
            Ok(())
        }
        _ => Err("a result where A's entry point has none, or none where it has one".into()),
    }
}

/// Why C has no program to run.
pub enum Unlowered {
    /// The lowering refuses the program: counted by its reason.
    Refused(String),
    /// The lowering made WGSL that is not the older program's interface, or not WGSL naga reads back: a failure.
    Unbuilt(String),
}

/// C's program for `iridia-exec`: `p` lowered for naga's own dialect and written as WGSL, read back by naga's front
/// end, with the older lift's interface and each of its resources bound to the global its binding plan names.
pub fn lower_c(old: &iridia::Program, p: &iridia_ir::Program) -> Result<iridia::Program, Unlowered> {
    // C runs where A does, in the executor of naga-path programs, which validates no 64-bit value
    let wide = |s: iridia_ir::Scalar| {
        matches!(
            s,
            iridia_ir::Scalar::U64 | iridia_ir::Scalar::I64 | iridia_ir::Scalar::F64
        )
    };
    if std::iter::once(&p.entry)
        .chain(p.patch.as_ref())
        .any(|f| f.values.iter().any(|d| wide(d.ty.scalar)))
    {
        return Err(Unlowered::Refused(
            "a 64-bit value, which the executor of naga-path programs does not validate".into(),
        ));
    }
    // a diagnostic: C's text from a file (`one`), to try a change to it
    let wgsl = match std::env::var("IR_GATE_C_WGSL") {
        Ok(path) => std::fs::read_to_string(path).unwrap(),
        // a module naga rejects after the preflight took the program is the lowering's fault, not a refusal
        Err(_) => iridia_naga::wgsl(p, &iridia_naga::Profile::NAGA).map_err(|e| {
            if e.0.starts_with("lowering bug") {
                Unlowered::Unbuilt(e.0)
            } else {
                Unlowered::Refused(e.0)
            }
        })?,
    };
    let unbuilt = |e: String| Unlowered::Unbuilt(e);
    let mut module = naga::front::wgsl::parse_str(&wgsl)
        .map_err(|e| unbuilt(format!("naga's WGSL front end: {}", e.emit_to_string(&wgsl))))?;
    pure_helpers::helpers(&mut module);
    normalise(&mut module, &old.module).map_err(unbuilt)?;
    let mut bindings = Vec::new();
    for r in &old.interface.resources {
        // the older DXIL lift's buffer of a view's hidden counter is the plan's counter of that view
        let (owner, suffix) = match r.provenance.counter_of {
            Some(view) => (
                old.interface
                    .resources
                    .iter()
                    .find(|v| v.id == view)
                    .ok_or_else(|| {
                        unbuilt(format!("{} counts a view the interface lacks", r.provenance.name))
                    })?,
                "_counter",
            ),
            None => (r, ""),
        };
        // a resource of the older lift with no D3D register (a descriptor heap) is placed by the plan after the registers
        // of its class: C cannot find its global by the older lift's binding
        let d = owner.provenance.d3d.ok_or_else(|| {
            Unlowered::Refused(format!(
                "resource {} of the older lift has no D3D register to name its plan global by",
                r.provenance.name
            ))
        })?;
        let prefix = match d.class {
            iridia::D3dClass::ConstantBuffer => "cb",
            iridia::D3dClass::ShaderResource => "t",
            iridia::D3dClass::Sampler => "s",
            iridia::D3dClass::UnorderedAccess => "u",
        };
        // the binding plan's names (`iridia_naga::BindingPlan::d3d`)
        let name = if d.space == 0 {
            format!("{prefix}{}{suffix}", d.register)
        } else {
            format!("{prefix}{}_space{}{suffix}", d.register, d.space)
        };
        let g = module
            .global_variables
            .iter()
            .find(|(_, g)| g.name.as_deref() == Some(&name))
            .map(|(h, _)| h)
            .ok_or_else(|| unbuilt(format!("no global {name}")))?;
        bindings.push((r.id, g));
    }
    let mut c = old.clone();
    c.module = module;
    c.bindings = bindings;
    Ok(c)
}
