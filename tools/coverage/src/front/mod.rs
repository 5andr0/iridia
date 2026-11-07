//! The bytecode front ends, as iridia's pipeline is: the one place a suite runner learns how a program is read,
//! lifted, written as WGSL and made runnable, so the runners follow the pipeline when it changes.
//!
//! On [`Path::Ir`], what the table grades, every bytecode goes through iridia's IR: DXBC (Shader Model 4 and 5) by
//! `iridia-dxbc`'s IR lift, DXIL (Shader Model 6) by `iridia-dxil`'s, Direct3D 9 token streams by `iridia-d3d9-ir`;
//! `iridia_passes::prepare` structures, promotes and simplifies it, and `iridia-exec` runs the IR. `iridia-naga` writes it as
//! WGSL, which naga's front end reads back: information beside the run, which decides none.
//!
//! On [`Path::Reference`], the lift the IR replaces, kept as the table's reference column until it is removed: DXBC
//! and DXIL lifted into naga modules, which `iridia-exec` runs and `iridia::emit` writes as WGSL; a Direct3D 9
//! program's WGSL read back by naga is what `iridia-exec` runs, and its IR is drawn again ([`Form::Ir`]) and held to
//! the WGSL's draw ([`d3d9`]).
pub mod d3d9;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use iridia_exec::geometry::GeometryShader;
use iridia_exec::mesh::MeshShader;
use iridia_exec::tessellation::TessellationShader;
use iridia_exec::{Executable, ProgramShader};

/// A program's bytecode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Bytecode {
    /// Direct3D 9 tokens (shader models 1 to 3).
    D3d9,
    /// DXBC (shader models 4 and 5).
    Dxbc,
    /// DXIL (shader model 6).
    Dxil,
}

impl Bytecode {
    /// The bytecode of a program's bytes: a DXBC container with a DXIL part is DXIL, any other container DXBC, and
    /// anything else Direct3D 9 tokens.
    pub fn of(bytes: &[u8]) -> Bytecode {
        if bytes.starts_with(b"DXBC") {
            match dxcontainer::chunks(bytes) {
                Ok(parts) if parts.iter().any(|c| &c.kind == b"DXIL" || &c.kind == b"ILDB") => Bytecode::Dxil,
                _ => Bytecode::Dxbc,
            }
        } else {
            Bytecode::D3d9
        }
    }

    /// Its name in reports.
    pub fn name(self) -> &'static str {
        match self {
            Bytecode::D3d9 => "d3d9",
            Bytecode::Dxbc => "dxbc",
            Bytecode::Dxil => "dxil",
        }
    }
}

/// Which lift a suite's programs go through.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Path {
    /// iridia's IR: each bytecode's IR lift (`iridia-dxbc`'s, `iridia-dxil`'s, `iridia-d3d9-ir`), the passes
    /// ([`iridia_passes::prepare`]), validated, and run in `iridia-exec` on the IR ([`Executable::from_ir`] and its
    /// kin). What the table grades.
    Ir,
    /// The lift it replaces, kept as the table's reference until it is removed: DXBC and DXIL lifted into naga
    /// modules by `iridia-dxbc` and `iridia-dxil`, run in `iridia-exec`; a Direct3D 9 program's WGSL read back by
    /// naga, its IR drawn again and held to it.
    Reference,
}

impl Path {
    /// Its name in reports.
    pub fn name(self) -> &'static str {
        match self {
            Path::Ir => "ir",
            Path::Reference => "reference",
        }
    }
}

/// The program `iridia-exec` runs.
#[derive(Clone, Debug)]
pub enum Code {
    /// A program on the IR ([`Path::Ir`]).
    Ir(Arc<iridia_ir::Program>),
    /// A naga-path program ([`Path::Reference`]): DXBC's or DXIL's as lifted, a Direct3D 9 program's WGSL read back.
    Naga(Arc<iridia::Program>),
}

impl Code {
    /// Its interface.
    pub fn interface(&self) -> &iridia::Interface {
        match self {
            Code::Ir(p) => &p.interface,
            Code::Naga(p) => &p.interface,
        }
    }

    /// The source's name for its target (`ps_3_0`).
    pub fn target(&self) -> &str {
        match self {
            Code::Ir(p) => &p.provenance.target,
            Code::Naga(p) => &p.provenance.target,
        }
    }

    /// It as `iridia-exec` runs a vertex or pixel program; with 32-bit denormals flushed where `flush`.
    fn shader(&self, flush: bool) -> Result<ProgramShader, String> {
        match self {
            Code::Ir(p) => {
                let mut p = iridia_ir::Program::clone(p);
                if flush {
                    p.float.f32 = iridia_ir::Denormals::Flush;
                }
                ProgramShader::from_ir(p)
            }
            Code::Naga(p) => {
                let mut p = iridia::Program::clone(p);
                if flush {
                    p.modes.flush_denormals = true;
                }
                ProgramShader::new(p)
            }
        }
        .map_err(|e| e.to_string())
    }

    /// It as `iridia-exec` runs any program.
    fn executable(&self) -> Result<Executable, String> {
        match self {
            Code::Ir(p) => Executable::from_ir(p),
            Code::Naga(p) => Executable::new(p),
        }
        .map_err(|e| e.to_string())
    }

    /// It as `iridia-exec` runs a geometry program.
    pub fn geometry(&self) -> Result<GeometryShader, String> {
        match self {
            Code::Ir(p) => GeometryShader::from_ir(p),
            Code::Naga(p) => GeometryShader::new(iridia::Program::clone(p)),
        }
    }

    /// It as `iridia-exec` runs a mesh program.
    pub fn mesh(&self) -> Result<MeshShader, String> {
        match self {
            Code::Ir(p) => MeshShader::from_ir(p),
            Code::Naga(p) => MeshShader::new(iridia::Program::clone(p)),
        }
        .map_err(|e| e.to_string())
    }

    /// `hull` and the `domain` program it feeds as `iridia-exec` runs them; both of one path.
    pub fn tessellation(hull: &Code, domain: &Code) -> Result<TessellationShader, String> {
        match (hull, domain) {
            (Code::Ir(h), Code::Ir(d)) => TessellationShader::from_ir(h, d),
            (Code::Naga(h), Code::Naga(d)) => {
                TessellationShader::new(iridia::Program::clone(h), iridia::Program::clone(d))
            }
            _ => Err("a hull and a domain program of different paths".into()),
        }
    }
}

/// How far a program got through the pipeline.
#[derive(Clone, Debug)]
pub struct Lifted {
    /// Its bytecode.
    pub bytecode: Bytecode,
    /// What `iridia-exec` runs.
    pub code: Code,
    /// Direct3D 9: the IR program in every form it is checked in.
    pub ir: Option<Arc<d3d9::Prepared>>,
    /// The WGSL written for it and read back by naga's front end, or why none was; on [`Path::Ir`] information only,
    /// which decides no run.
    pub wgsl: Result<(), String>,
}

impl Lifted {
    /// Whether the program leaves it open whether 32-bit denormals are flushed or kept, so a run may do either:
    /// Direct3D 9's (its IR's `Denormals::Any`). DXBC and DXIL programs state theirs.
    pub fn denormals_open(&self) -> bool {
        let float = match (&self.code, &self.ir) {
            (Code::Ir(p), _) => p.float,
            (Code::Naga(_), Some(p)) => p.ir.float,
            (Code::Naga(_), None) => return false,
        };
        float.f32 == iridia_ir::Denormals::Any
    }
}

/// Why a program did not get as far as running.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stop {
    /// The lifter refused it.
    Refused(String),
    /// It lifted and a later stage refused it (a pass, validation; on [`Path::Reference`], the WGSL that a Direct3D 9
    /// program runs as).
    Failed(String),
}

impl Stop {
    /// The message.
    pub fn message(&self) -> &str {
        match self {
            Stop::Refused(m) | Stop::Failed(m) => m,
        }
    }
}

/// That naga's WGSL front end reads `wgsl` back and its validator accepts it.
pub fn read_back(wgsl: &str) -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(wgsl).map_err(|e| format!("naga's WGSL front end: {e}"))?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| format!("naga's validator: {e}"))?;
    Ok(())
}

/// A program lifted onto the IR, as `iridia-exec` runs it: through the passes every output and executor takes
/// ([`iridia_passes::prepare`]), and validated.
fn passed(mut p: iridia_ir::Program) -> Result<iridia_ir::Program, Stop> {
    iridia_passes::prepare(&mut p).map_err(Stop::Failed)?;
    if let Err(e) = iridia_ir::validate(&p) {
        return Err(Stop::Failed(format!("IR validation: {}", e[0])));
    }
    Ok(p)
}

/// Lift a program of any bytecode by `path`; a Direct3D 9 program that reads pipeline state at Direct3D 9's defaults,
/// with what makes it again for other state.
pub fn lift(bytes: &[u8], path: Path) -> Result<(Lifted, Option<d3d9::Stateful>), Stop> {
    let bytecode = Bytecode::of(bytes);
    if bytecode == Bytecode::D3d9 {
        return d3d9::lift(bytes, path);
    }
    let lifted = match path {
        Path::Ir => {
            let p = match bytecode {
                Bytecode::Dxbc => iridia_dxbc::ir::lift(bytes),
                _ => iridia_dxil::ir::lift(bytes),
            }
            .map_err(|e| Stop::Refused(e.to_string()))?;
            let p = passed(p)?;
            let wgsl = iridia_naga::wgsl(&p, &iridia_naga::Profile::BROWSER)
                .map_err(|e| format!("WGSL: {e}"))
                .and_then(|text| read_back(&text));
            Lifted {
                bytecode,
                code: Code::Ir(Arc::new(p)),
                ir: None,
                wgsl,
            }
        }
        Path::Reference => {
            let program = match bytecode {
                Bytecode::Dxbc => iridia_dxbc::lift(bytes),
                _ => iridia_dxil::lift(bytes),
            }
            .map_err(|e| Stop::Refused(e.to_string()))?;
            let wgsl = iridia::emit::wgsl(&program)
                .map_err(|e| format!("WGSL: {e}"))
                .and_then(|text| read_back(&text));
            Lifted {
                bytecode,
                code: Code::Naga(Arc::new(program)),
                ir: None,
                wgsl,
            }
        }
    };
    Ok((lifted, None))
}

/// A DXIL program lifted onto the IR and structured: on [`Path::Reference`], what an amplification program and the
/// mesh program it launches run as, which the naga-path lift does not take.
fn lift_ir(bytes: &[u8]) -> Result<iridia_ir::Program, Stop> {
    if Bytecode::of(bytes) != Bytecode::Dxil {
        return Err(Stop::Refused(
            "an amplification pipeline's program that is not DXIL".into(),
        ));
    }
    let mut p = iridia_dxil::ir::lift(bytes).map_err(|e| Stop::Refused(e.to_string()))?;
    for f in std::iter::once(&mut p.entry).chain(p.patch.as_mut()) {
        if matches!(f.body, iridia_ir::Body::Cfg(_)) {
            iridia_passes::structurize(f).map_err(Stop::Failed)?;
            iridia_passes::break_flags(f).map_err(Stop::Failed)?;
        }
    }
    Ok(p)
}

/// A program lifted and made runnable.
#[derive(Debug)]
pub struct Ready {
    /// How far it got.
    pub lifted: Lifted,
    /// It as `iridia-exec` runs a vertex or pixel program, or why not.
    pub shader: Result<ProgramShader, String>,
    /// Where the program leaves denormals open ([`Lifted::denormals_open`]): [`Ready::shader`] with 32-bit denormals
    /// flushed, which `shader` keeps.
    pub flushed: Option<Result<ProgramShader, String>>,
    /// On [`Path::Reference`], a Direct3D 9 program's IR as `iridia-exec` runs it ([`ProgramShader::from_ir`]), whose
    /// draw is held to its WGSL's, or why not.
    pub ir_shader: Option<Result<ProgramShader, String>>,
    /// [`Ready::ir_shader`] with 32-bit denormals flushed, where the program leaves them open.
    pub ir_flushed: Option<Result<ProgramShader, String>>,
    /// It as `iridia-exec` runs any program, or why not.
    pub executable: Result<Executable, String>,
}

/// The form of a program a draw runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// What its path runs ([`Lifted::code`]).
    Lifted,
    /// On [`Path::Reference`], a Direct3D 9 program's IR, drawn again and held to its WGSL's draw.
    Ir,
}

impl Ready {
    /// A lifted program made runnable in every form it has.
    fn of(lifted: Lifted) -> Self {
        let shader = lifted.code.shader(false);
        let flushed = lifted.denormals_open().then(|| lifted.code.shader(true));
        let executable = lifted.code.executable();
        // beside a Direct3D 9 program's WGSL, its IR in the register view the WGSL runs in
        let ir = match &lifted.code {
            Code::Naga(_) => lifted
                .ir
                .as_ref()
                .map(|p| d3d9::registered(&p.ir).map(|ir| Code::Ir(Arc::new(ir)))),
            Code::Ir(_) => None,
        };
        let ir_shader = ir
            .as_ref()
            .map(|c| c.as_ref().map_err(Clone::clone)?.shader(false));
        let ir_flushed = ir.filter(|_| lifted.denormals_open()).map(|c| c?.shader(true));
        Ready {
            lifted,
            shader,
            flushed,
            ir_shader,
            ir_flushed,
            executable,
        }
    }

    /// The vertex or pixel program `iridia-exec` runs in `form`, its denormals flushed where `flushed` and the
    /// program leaves them open; `None` where the program has no such form.
    pub fn shader(&self, form: Form, flushed: bool) -> Option<&Result<ProgramShader, String>> {
        let (kept, flush) = match form {
            Form::Lifted => (Some(&self.shader), self.flushed.as_ref()),
            Form::Ir => (self.ir_shader.as_ref(), self.ir_flushed.as_ref()),
        };
        if flushed { flush.or(kept) } else { kept }
    }
}

// The stop of a lock another thread panicked holding.
fn poisoned<T>(_: T) -> Stop {
    Stop::Failed("a poisoned lock".into())
}

/// A program ready to run, or why it stopped.
pub type Lift = Result<Arc<Ready>, Stop>;

/// A Direct3D 9 program that reads pipeline state, by its name and the values of the state it reads.
type Specialised = (String, Vec<[u64; 4]>);

/// Each program lifted so far, by name: it, ready to run, or why it stopped.
pub type Lifts = BTreeMap<String, Lift>;

type Source<'f> = Box<dyn Fn(&str) -> Result<Vec<u8>, String> + Send + Sync + 'f>;

/// A suite's programs by name, lifted once and shared by the runs that use them.
pub struct Programs<'f> {
    bytes: Source<'f>,
    /// The lift they go through.
    path: Path,
    /// Each program at Direct3D 9's default state ([`d3d9::default_state`]) where it reads any.
    lifted: Mutex<Lifts>,
    /// The Direct3D 9 programs that read pipeline state, by name.
    stateful: Mutex<BTreeMap<String, Arc<d3d9::Stateful>>>,
    /// Those programs made for other state, by name and the values of the state they read.
    specialised: Mutex<BTreeMap<Specialised, Lift>>,
}

impl std::fmt::Debug for Programs<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Programs").finish_non_exhaustive()
    }
}

impl<'f> Programs<'f> {
    /// The programs `bytes` finds by name, to go through `path`.
    pub fn new(path: Path, bytes: impl Fn(&str) -> Result<Vec<u8>, String> + Send + Sync + 'f) -> Self {
        Programs {
            bytes: Box::new(bytes),
            path,
            lifted: Mutex::new(BTreeMap::new()),
            stateful: Mutex::new(BTreeMap::new()),
            specialised: Mutex::new(BTreeMap::new()),
        }
    }

    /// Program `name`'s bytes.
    pub fn bytes(&self, name: &str) -> Result<Vec<u8>, String> {
        (self.bytes)(name)
    }

    /// The lift they go through.
    pub fn path(&self) -> Path {
        self.path
    }

    /// Program `name` on the IR, for an amplification pipeline: on [`Path::Ir`] as [`get`](Self::get) lifts it; on
    /// [`Path::Reference`], whose naga-path lift takes no amplification program, lifted onto the IR and structured.
    pub fn ir(&self, name: &str) -> Result<Arc<iridia_ir::Program>, Stop> {
        match self.path {
            Path::Ir => match &self.get(name)?.lifted.code {
                Code::Ir(p) => Ok(Arc::clone(p)),
                Code::Naga(_) => Err(Stop::Failed(format!("{name} is not on the IR"))),
            },
            Path::Reference => {
                let bytes = (self.bytes)(name).map_err(Stop::Failed)?;
                lift_ir(&bytes).map(Arc::new)
            }
        }
    }

    /// Program `name`, lifted; at Direct3D 9's default state where it reads any.
    pub fn get(&self, name: &str) -> Result<Arc<Ready>, Stop> {
        if let Some(done) = self.lifted.lock().map_err(poisoned)?.get(name) {
            return done.clone();
        }
        let bytes = (self.bytes)(name).map_err(Stop::Failed)?;
        let done = lift(&bytes, self.path).map(|(lifted, stateful)| {
            if let Some(stateful) = stateful
                && let Ok(mut known) = self.stateful.lock()
            {
                known.insert(name.to_owned(), Arc::new(stateful));
            }
            Arc::new(Ready::of(lifted))
        });
        self.lifted
            .lock()
            .map_err(poisoned)?
            .insert(name.to_owned(), done.clone());
        done
    }

    /// Program `name` for a draw whose pipeline state `state` gives: a Direct3D 9 program that reads state is
    /// specialised by it ([`d3d9::Stateful`]); any other is [`get`](Self::get)'s.
    pub fn get_in(&self, name: &str, state: d3d9::StateOf<'_>) -> Result<Arc<Ready>, Stop> {
        let base = self.get(name)?;
        let Some(stateful) = self.stateful.lock().map_err(poisoned)?.get(name).cloned() else {
            return Ok(base);
        };
        let values = stateful.values(state);
        if values == stateful.values(&d3d9::default_state) {
            return Ok(base);
        }
        let key = (name.to_owned(), values);
        if let Some(done) = self.specialised.lock().map_err(poisoned)?.get(&key) {
            return done.clone();
        }
        let done = stateful.lift(&key.1).map(|lifted| Arc::new(Ready::of(lifted)));
        self.specialised
            .lock()
            .map_err(poisoned)?
            .insert(key, done.clone());
        done
    }

    /// Every program lifted so far.
    pub fn lifted(&self) -> Lifts {
        self.lifted.lock().map(|l| l.clone()).unwrap_or_default()
    }
}
