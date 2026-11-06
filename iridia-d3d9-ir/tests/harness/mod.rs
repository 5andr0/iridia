//! What the tests that run lifted programs share: the forms a program is checked in (as lifted, as a control flow
//! graph and structured again, and as WGSL read back by naga's front end), the runs of the IR in `iridia-exec-ir` and of
//! a naga program in `iridia-exec` in one register view, and a texture model both executors sample alike.
#![allow(dead_code, reason = "each test uses the part it needs")]

use std::{cell::RefCell, collections::BTreeMap};

use iridia_exec::{Dim, Execution, ImageLod, ImageSample, PixelShader, ProgramShader, Sample, Sampler};
use iridia_ir::interface::{D3dSignature, Interface, ResourceId, Stage, SystemValue};

/// The lanes a write mask names, in order.
pub fn lanes_of(mask: u8) -> Vec<usize> {
    (0..4).filter(|n| mask & (1 << n) != 0).collect()
}

/// That `f` prints as text that parses back to what prints the same.
pub fn round_trip(f: &iridia_ir::Function) -> Result<(), String> {
    let text = iridia_ir::text::print(f);
    match iridia_ir::text::parse(&text) {
        Ok(back) if iridia_ir::text::print(&back) == text => Ok(()),
        Ok(_) => Err("IR text: printed differently after parsing".into()),
        Err(e) => Err(format!("IR text: {e}")),
    }
}

/// The `If`, `Loop` and `Block` constructs of a region and those inside it.
pub fn constructs(r: &iridia_ir::Region) -> [usize; 3] {
    let mut n = [0; 3];
    let mut add = |m: [usize; 3]| (0..3).for_each(|i| n[i] += m[i]);
    for node in &r.nodes {
        match node {
            iridia_ir::Node::If { accept, reject, .. } => {
                add([1, 0, 0]);
                add(constructs(accept));
                add(constructs(reject));
            }
            iridia_ir::Node::Loop { body, .. } => {
                add([0, 1, 0]);
                add(constructs(body));
            }
            iridia_ir::Node::Block { body, .. } => {
                add([0, 0, 1]);
                add(constructs(body));
            }
            _ => {}
        }
    }
    n
}

/// A lifted program in every form it is checked in.
pub struct Prepared {
    /// As lifted, promoted and simplified.
    pub ir: iridia_ir::Program,
    /// As a control flow graph, by `flatten`.
    pub graph: iridia_ir::Program,
    /// The graph structured again, by `structurize` (which promotes) and `simplify`.
    pub restructured: iridia_ir::Program,
    /// The WGSL for browsers, read back by naga's front end, for `iridia-exec`.
    pub wgsl: ProgramShader,
    /// The interface `wgsl` is run by: the program's, with each varying in a register of its own (see [`from_wgsl`]).
    pub wgsl_interface: Interface,
}

/// Promote, simplify, validate and print a lifted program, make it a graph and structure it again (each form
/// validated and printed), and write it as WGSL; an error is the first check it fails.
pub fn prepare(mut ir: iridia_ir::Program) -> Result<Prepared, String> {
    iridia_passes::promote(&mut ir.entry).map_err(|e| format!("promote: {e}"))?;
    iridia_passes::simplify(&mut ir.entry);
    if let Err(e) = iridia_ir::validate(&ir) {
        return Err(format!("IR validation: {}", e[0]));
    }
    round_trip(&ir.entry)?;
    // the graph form and back, as a graph-shaped source takes it
    let mut graph = ir.clone();
    iridia_passes::flatten(&mut graph.entry).map_err(|e| format!("flatten: {e}"))?;
    if let Err(e) = iridia_ir::validate(&graph) {
        return Err(format!("IR validation of the graph: {}", e[0]));
    }
    round_trip(&graph.entry)?;
    let mut restructured = graph.clone();
    iridia_passes::structurize(&mut restructured.entry).map_err(|e| format!("structurize: {e}"))?;
    iridia_passes::simplify(&mut restructured.entry);
    if let Err(e) = iridia_ir::validate(&restructured) {
        return Err(format!("IR validation of the restructured graph: {}", e[0]));
    }
    round_trip(&restructured.entry)?;
    let wgsl = iridia_naga::wgsl(&ir, &iridia_naga::Profile::BROWSER).map_err(|e| format!("WGSL: {e}"))?;
    let program = from_wgsl(&ir, &wgsl)?;
    let wgsl_interface = program.interface.clone();
    let wgsl =
        ProgramShader::new(program).map_err(|e| format!("WGSL program not compiled for iridia-exec: {e}"))?;
    Ok(Prepared {
        ir,
        graph,
        restructured,
        wgsl,
        wgsl_interface,
    })
}

/// The program the WGSL is, for `iridia-exec`: the IR program's interface over the module naga's front end reads
/// back, each resource bound to the global at the binding the plan gives it. `iridia-exec` carries varyings in a
/// register view by their D3D registers, where shader model 1 and 2 put two in one (`v0` and `t0`, `oD0` and `oT0`)
/// and `vPos` and `vFace` have none: each input and each colour or varying output is put in the register of its slot,
/// its lanes as before. What WGSL text cannot say is put back from the interface: the entry point's argument and result
/// member names (naga's namer writes `COLOR0_` for `COLOR0`) and bindings (WGSL cannot say a varying has no
/// interpolation, so naga's front end supplies the default).
pub fn from_wgsl(p: &iridia_ir::Program, wgsl: &str) -> Result<iridia::Program, String> {
    let mut module = naga::front::wgsl::parse_str(wgsl).map_err(|e| format!("naga's WGSL front end: {e}"))?;
    let plan = iridia_naga::BindingPlan::d3d(&p.interface).map_err(|e| e.to_string())?;
    let mut bindings = Vec::new();
    for b in &plan.bindings {
        let at = naga::ResourceBinding {
            group: b.group,
            binding: b.binding,
        };
        let g = module
            .global_variables
            .iter()
            .find(|(_, g)| g.binding.as_ref() == Some(&at))
            .map(|(h, _)| h)
            .ok_or_else(|| format!("no global at group {} binding {}", b.group, b.binding))?;
        bindings.push((b.resource, g));
    }
    let mut interface = p.interface.clone();
    for (n, v) in interface.inputs.iter_mut().enumerate() {
        v.provenance.d3d = Some(D3dSignature {
            register: u32::try_from(n).unwrap(),
            mask: v
                .provenance
                .d3d
                .map_or(if v.value.width == 1 { 1 } else { 0b1111 }, |d| d.mask),
            read_write_mask: 0,
        });
    }
    for (n, v) in interface.outputs.iter_mut().enumerate() {
        if let (Some(d), None | Some(SystemValue::Target(_))) = (&mut v.provenance.d3d, v.system) {
            d.register = u32::try_from(n).unwrap();
        }
    }
    let io = iridia::entry_io(&interface)?;
    let ep = module
        .entry_points
        .first_mut()
        .ok_or("the WGSL has no entry point")?;
    let entry = ep.name.clone();
    if ep.function.arguments.len() != io.arguments.len() {
        return Err("the WGSL entry point's arguments are not the interface's".into());
    }
    for (a, item) in ep.function.arguments.iter_mut().zip(&io.arguments) {
        a.name = Some(item.name.clone());
        a.binding = Some(item.binding.clone());
    }
    if let Some(result) = &ep.function.result {
        let mut t = module.types[result.ty].clone();
        let naga::TypeInner::Struct { members, .. } = &mut t.inner else {
            return Err("the WGSL entry point's result is not a structure".into());
        };
        if members.len() != io.results.len() {
            return Err("the WGSL entry point's results are not the interface's".into());
        }
        for (m, item) in members.iter_mut().zip(&io.results) {
            m.name = Some(item.name.clone());
            m.binding = Some(item.binding.clone());
        }
        module.types.replace(result.ty, t);
    }
    let program = iridia::Program {
        module,
        stage: p.stage,
        entry,
        interface,
        bindings,
        conventions: iridia::Conventions::D3D,
        modes: iridia::D3dModes {
            min_max_nan: true,
            saturate_nan: true,
            ..Default::default()
        },
        provenance: iridia::Provenance {
            format: iridia::ShaderFormat::D3d9,
            target: p.provenance.target.clone(),
            lifter: "iridia-d3d9-ir through WGSL".into(),
            lifter_version: 0,
            source_hash: None,
            instruction_count: None,
        },
        notes: Vec::new(),
    };
    Ok(program)
}

/// One invocation's results: each colour and varying output's four lanes by interface slot (lanes its mask does not
/// name 0), the clip position, the outputs `iridia-exec`'s register view does not carry (depth, coverage, stencil) by
/// slot where the run sees them, and whether it was killed.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct View {
    pub outputs: BTreeMap<usize, [u32; 4]>,
    pub position: Option<[u32; 4]>,
    pub other: BTreeMap<usize, [u32; 4]>,
    pub killed: bool,
}

impl View {
    /// Alike where both runs see them, which `other` is not: the same bits in every lane, or a NaN in both. IEEE 754
    /// states a NaN's sign and payload only for the sign operations, and two executors keep different ones (through
    /// `fract` of a negated NaN, say). A killed pixel writes nothing, so two killed runs are alike whatever their
    /// outputs: `iridia-exec-ir` keeps those stored before the kill, and `iridia-exec` a helper's later stores too.
    pub fn alike(&self, other: &View) -> bool {
        let lanes = |a: &[u32; 4], b: &[u32; 4]| {
            a.iter()
                .zip(b)
                .all(|(&x, &y)| x == y || f32::from_bits(x).is_nan() && f32::from_bits(y).is_nan())
        };
        if self.killed && other.killed && self.position.is_none() && other.position.is_none() {
            return true;
        }
        self.killed == other.killed
            && self.outputs.len() == other.outputs.len()
            && self
                .outputs
                .iter()
                .zip(&other.outputs)
                .all(|((r, a), (s, b))| r == s && lanes(a, b))
            && match (&self.position, &other.position) {
                (Some(a), Some(b)) => lanes(a, b),
                (a, b) => a == b,
            }
    }

    /// Equal where both runs see them, as `alike` but bit for bit.
    pub fn equal(&self, other: &View) -> bool {
        self.killed == other.killed && self.outputs == other.outputs && self.position == other.position
    }
}

/// What a run of the IR binds besides its inputs: constant banks and pipeline state by resource, a sampler, and the
/// denormal rule where the program leaves it open.
pub struct Bound<'a> {
    pub buffers: &'a [(ResourceId, &'a [[u32; 4]])],
    pub state: &'a [(ResourceId, [u32; 4])],
    pub sampler: &'a mut dyn FnMut(iridia_exec_ir::Sample) -> Result<[f32; 4], String>,
    pub denormals: iridia_ir::Denormals,
}

/// A run of the IR in `iridia-exec-ir`, its inputs by slot (each varying's lanes packed from lane 0).
pub fn run_ir(p: &iridia_ir::Program, inputs: &[[u32; 4]], bound: &mut Bound<'_>) -> Result<View, String> {
    let interface = &p.interface;
    let run = iridia_exec_ir::run(
        p,
        &mut iridia_exec_ir::Bound {
            inputs,
            buffers: bound.buffers,
            state: bound.state,
            sampler: &mut *bound.sampler,
            denormals: bound.denormals,
        },
    )?;
    ir_view(interface, run)
}

/// A quad's runs of the IR in `iridia-exec-ir` (lanes 0 and 1 the top row), each lane's inputs by slot, the lanes
/// sharing `bound`'s resources and sampler.
pub fn run_ir_quad(
    p: &iridia_ir::Program,
    inputs: [&[[u32; 4]]; 4],
    bound: &mut Bound<'_>,
) -> Result<[View; 4], String> {
    let shared = RefCell::new(&mut *bound.sampler);
    let answer = |s: iridia_exec_ir::Sample| (shared.borrow_mut())(s);
    let [mut a, mut b, mut c, mut d] = [answer; 4];
    let [i0, i1, i2, i3] = inputs;
    let lane = |inputs, sampler| iridia_exec_ir::Bound {
        inputs,
        buffers: bound.buffers,
        state: bound.state,
        sampler,
        denormals: bound.denormals,
    };
    let (mut b0, mut b1, mut b2, mut b3) = (
        lane(i0, &mut a),
        lane(i1, &mut b),
        lane(i2, &mut c),
        lane(i3, &mut d),
    );
    let runs = iridia_exec_ir::run_quad(p, [&mut b0, &mut b1, &mut b2, &mut b3])?;
    let [r0, r1, r2, r3] = runs;
    Ok([
        ir_view(&p.interface, r0)?,
        ir_view(&p.interface, r1)?,
        ir_view(&p.interface, r2)?,
        ir_view(&p.interface, r3)?,
    ])
}

/// The view of a run of the IR.
fn ir_view(interface: &Interface, run: iridia_exec_ir::Run) -> Result<View, String> {
    let (mut outputs, mut other, mut position) = (BTreeMap::new(), BTreeMap::new(), None);
    for (slot, v) in interface.outputs.iter().enumerate() {
        let got = run.outputs[slot].unwrap_or([0; 4]);
        match v.system {
            Some(SystemValue::ClipPosition) => position = Some(got),
            None | Some(SystemValue::Target(_)) => {
                let d = v.provenance.d3d.ok_or("an output with no register")?;
                let mut lanes = [0u32; 4];
                for (k, l) in lanes_of(d.mask).into_iter().enumerate() {
                    lanes[l] = got[k];
                }
                outputs.insert(slot, lanes);
            }
            Some(_) => {
                other.insert(slot, got);
            }
        }
    }
    Ok(View {
        outputs,
        position,
        other,
        killed: run.killed,
    })
}

/// A run of a naga program in `iridia-exec`, its inputs in the register view, its outputs read from it by the registers
/// and lanes `interface` gives them.
pub fn run_naga(
    shader: &ProgramShader,
    interface: &Interface,
    inputs: &[[u32; 4]],
    buffers: &[(u32, &[[u32; 4]])],
    sampler: &mut dyn Sampler,
) -> Result<View, String> {
    let (e, position) = if interface.stage == Stage::Vertex {
        let (e, p) = shader
            .execute_vertex(inputs, buffers, sampler)
            .map_err(|e| e.to_string())?;
        (e, Some(p))
    } else {
        (
            shader
                .execute(inputs, buffers, sampler)
                .map_err(|e| e.to_string())?,
            None,
        )
    };
    naga_view(interface, e, position)
}

/// A quad's runs of a naga program in `iridia-exec` (lanes 0 and 1 the top row), as [`run_naga`] runs one.
pub fn run_naga_quad(
    shader: &ProgramShader,
    interface: &Interface,
    inputs: [&[[u32; 4]]; 4],
    buffers: &[(u32, &[[u32; 4]])],
    sampler: &mut dyn Sampler,
) -> Result<[View; 4], String> {
    let [e0, e1, e2, e3] = shader
        .execute_quad(inputs, buffers, sampler)
        .map_err(|e| e.to_string())?;
    Ok([
        naga_view(interface, e0, None)?,
        naga_view(interface, e1, None)?,
        naga_view(interface, e2, None)?,
        naga_view(interface, e3, None)?,
    ])
}

/// The view of a run of a naga program, its outputs read from the register view.
fn naga_view(interface: &Interface, e: Execution, position: Option<[u32; 4]>) -> Result<View, String> {
    let mut outputs = BTreeMap::new();
    for (slot, v) in interface.outputs.iter().enumerate() {
        if !matches!(v.system, None | Some(SystemValue::Target(_))) {
            continue;
        }
        let d = v.provenance.d3d.ok_or("an output with no register")?;
        let got = e.outputs.get(d.register as usize).copied().unwrap_or([0; 4]);
        let mut lanes = [0u32; 4];
        for l in lanes_of(d.mask) {
            lanes[l] = got[l];
        }
        outputs.insert(slot, lanes);
    }
    Ok(View {
        outputs,
        position,
        other: BTreeMap::new(),
        killed: e.discarded,
    })
}

impl Prepared {
    /// The register view of inputs given by slot: each varying's packed lanes put in the lanes its mask names, in the
    /// register of its slot.
    pub fn register_view(&self, slots: &[[u32; 4]]) -> Vec<[u32; 4]> {
        let mut registers = vec![[0u32; 4]; slots.len().max(1)];
        for (n, (v, packed)) in self.wgsl_interface.inputs.iter().zip(slots).enumerate() {
            for (k, l) in lanes_of(v.provenance.d3d.map_or(0, |d| d.mask))
                .into_iter()
                .enumerate()
            {
                registers[n][l] = packed[k];
            }
        }
        registers
    }

    /// The IR's run, which its graph and the graph structured again must match bit for bit (no operation differs
    /// between them, only the control around them); then, where `bound` keeps denormals as `iridia-exec` does, the
    /// WGSL's, which must be alike. `banks` are the constant banks by `iridia-exec` register, `sampler` its sampler.
    pub fn run(
        &self,
        slots: &[[u32; 4]],
        bound: &mut Bound<'_>,
        banks: &[(u32, &[[u32; 4]])],
        sampler: &mut dyn Sampler,
    ) -> Result<View, String> {
        let b = run_ir(&self.ir, slots, bound)?;
        for (form, program) in [("graph", &self.graph), ("restructured graph", &self.restructured)] {
            let got = run_ir(program, slots, bound).map_err(|e| format!("the {form}: {e}"))?;
            if got != b {
                return Err(format!("the IR and its {form} disagree: {b:x?} and {got:x?}"));
            }
        }
        if bound.denormals == iridia_ir::Denormals::Flush {
            return Ok(b);
        }
        let c = run_naga(
            &self.wgsl,
            &self.wgsl_interface,
            &self.register_view(slots),
            banks,
            sampler,
        )
        .map_err(|e| format!("iridia-exec: {e}"))?;
        if !b.alike(&c) {
            return Err(format!("the IR and the WGSL disagree: {b:x?} and {c:x?}"));
        }
        Ok(b)
    }

    /// [`Prepared::run`] for a pixel program's 2 x 2 quad (lanes 0 and 1 the top row), each lane's inputs by slot:
    /// every form's four lanes must match the IR's as there.
    pub fn run_quad(
        &self,
        slots: [&[[u32; 4]]; 4],
        bound: &mut Bound<'_>,
        banks: &[(u32, &[[u32; 4]])],
        sampler: &mut dyn Sampler,
    ) -> Result<[View; 4], String> {
        let b = run_ir_quad(&self.ir, slots, bound)?;
        for (form, program) in [("graph", &self.graph), ("restructured graph", &self.restructured)] {
            let got = run_ir_quad(program, slots, bound).map_err(|e| format!("the {form}: {e}"))?;
            if got != b {
                return Err(format!("the IR and its {form} disagree: {b:x?} and {got:x?}"));
            }
        }
        if bound.denormals == iridia_ir::Denormals::Flush {
            return Ok(b);
        }
        let registers = slots.map(|s| self.register_view(s));
        let c = run_naga_quad(
            &self.wgsl,
            &self.wgsl_interface,
            [&registers[0], &registers[1], &registers[2], &registers[3]],
            banks,
            sampler,
        )
        .map_err(|e| format!("iridia-exec: {e}"))?;
        for (l, (b, c)) in b.iter().zip(&c).enumerate() {
            if !b.alike(c) {
                return Err(format!(
                    "the IR and the WGSL disagree in lane {l}: {b:x?} and {c:x?}"
                ));
            }
        }
        Ok(b)
    }
}

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// A seeded stream of test values.
pub struct Rng(u64);

impl Rng {
    pub fn of(name: &str) -> Rng {
        Rng(mix(name.bytes().fold(0u64, |h, b| mix(h ^ u64::from(b)))))
    }
    pub fn next(&mut self) -> u64 {
        self.0 = mix(self.0);
        self.0
    }
    /// A float in -4..4, or one of the values that find edge cases, one time in eight.
    pub fn float(&mut self) -> f32 {
        const SPECIAL: [f32; 10] = [0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 1e-3, 100.0, -0.25, 3.0];
        let r = self.next();
        if r.is_multiple_of(8) {
            SPECIAL[(r >> 8) as usize % SPECIAL.len()]
        } else {
            ((r >> 11) as f64 / (1u64 << 53) as f64 * 8.0 - 4.0) as f32
        }
    }
}

/// A texel from what the request names: the texture's register, the coordinate's bits and the level (the gradients'
/// lanes of the coordinate's, where it states them).
pub fn texel(texture: u32, coord: &[f32], lod: ImageLod) -> [f32; 4] {
    let mut h = mix(u64::from(texture) ^ 0x5151);
    for c in coord {
        h = mix(h ^ u64::from(c.to_bits()));
    }
    let gradients = |h: u64, g: &[[f32; 3]; 2]| {
        g.iter()
            .flat_map(|d| &d[..coord.len()])
            .fold(h, |h, x| mix(h ^ u64::from(x.to_bits())))
    };
    h = match lod {
        ImageLod::Implicit => mix(h ^ 1),
        ImageLod::Bias(b) => mix(h ^ 2 ^ (u64::from(b.to_bits()) << 8)),
        ImageLod::Level(l) => mix(h ^ 3 ^ (u64::from(l.to_bits()) << 8)),
        ImageLod::Gradients(g) => gradients(mix(h ^ 4), &g),
        ImageLod::GradientsBias(g, b) => gradients(mix(h ^ 5 ^ (u64::from(b.to_bits()) << 8)), &g),
    };
    let mut out = [0.0; 4];
    for (i, o) in out.iter_mut().enumerate() {
        *o = ((mix(h ^ i as u64) >> 40) as f32) / (1u64 << 24) as f32;
    }
    out
}

/// The texel model for `iridia-exec`.
pub struct Texels;

impl Sampler for Texels {
    fn sample(&mut self, r: Sample) -> Result<[f32; 4], String> {
        Ok(texel(r.texture, &r.uv, r.lod.into()))
    }
    fn sample_image(&mut self, r: ImageSample<u32>) -> Result<[f32; 4], String> {
        if r.gather.is_some() || r.depth_ref.is_some() || r.offset != [0; 3] || r.layer.is_some() {
            return Err("an unexpected sample".into());
        }
        let lanes = match r.dim {
            Dim::D1 => 1,
            Dim::D2 => 2,
            Dim::D3 | Dim::Cube => 3,
        };
        Ok(texel(r.texture, &r.coord[..lanes], r.lod))
    }
}

/// The texel model for `iridia-exec-ir`, the texture named by its register in `interface`.
pub fn ir_texels(
    interface: &Interface,
) -> impl FnMut(iridia_exec_ir::Sample) -> Result<[f32; 4], String> + '_ {
    move |s: iridia_exec_ir::Sample| {
        let register = interface
            .resources
            .iter()
            .find(|r| r.id == s.texture)
            .and_then(|r| r.provenance.d3d)
            .map_or(u32::MAX, |d| d.register);
        let lod = match s.lod {
            iridia_exec_ir::Lod::Auto => ImageLod::Implicit,
            iridia_exec_ir::Lod::Bias(b) => ImageLod::Bias(b),
            iridia_exec_ir::Lod::Exact(l) => ImageLod::Level(l),
            iridia_exec_ir::Lod::Grad(dx, dy) => ImageLod::Gradients([dx, dy]),
            iridia_exec_ir::Lod::GradBias(dx, dy, b) => ImageLod::GradientsBias([dx, dy], b),
        };
        Ok(texel(register, &s.coord[..usize::from(s.lanes)], lod))
    }
}

/// No textures, for `iridia-exec`.
pub struct NoTextures;

impl Sampler for NoTextures {
    fn sample(&mut self, _: Sample) -> Result<[f32; 4], String> {
        Err("a texture sample".into())
    }
}
