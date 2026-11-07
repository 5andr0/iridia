//! The bytecode front ends, as iridia's public pipeline is today: the one place a suite runner learns how a program is
//! read, lifted, written as WGSL and made runnable, so the runners follow the pipeline when it changes.
//!
//! - Direct3D 9 token streams: `iridia-d3d9-ir` lifts into iridia's IR, `iridia-passes` promotes and simplifies it,
//!   `iridia-naga` writes WGSL, and the WGSL read back by naga's front end is what `iridia-exec` runs; the IR itself
//!   runs in `iridia-exec-ir` beside it ([`d3d9`]), and in `iridia-exec` too ([`Form::Ir`]), where a runner draws
//!   with it again and holds it to the WGSL's draw.
//! - DXBC (Shader Model 4 and 5) and DXIL (Shader Model 6): `iridia-dxbc` and `iridia-dxil` lift into naga modules,
//!   which `iridia-exec` runs and `iridia::emit` writes as WGSL, read back by naga's front end to check it.
pub mod d3d9;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

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

/// How far a program got through the pipeline.
#[derive(Clone, Debug)]
pub struct Lifted {
    /// Its bytecode.
    pub bytecode: Bytecode,
    /// What `iridia-exec` runs: the lifted program (DXBC, DXIL) or the WGSL read back (Direct3D 9).
    pub program: Arc<iridia::Program>,
    /// Direct3D 9: the IR program in every form it is checked in.
    pub ir: Option<Arc<d3d9::Prepared>>,
    /// The WGSL written for it, or why none was.
    pub wgsl: Result<(), String>,
}

impl Lifted {
    /// Whether the program leaves it open whether 32-bit denormals are flushed or kept, so a run may do either:
    /// Direct3D 9's (its IR's `Denormals::Any`). DXBC and DXIL programs state theirs ([`iridia::D3dModes`]).
    pub fn denormals_open(&self) -> bool {
        self.ir
            .as_ref()
            .is_some_and(|p| p.ir.float.f32 == iridia_ir::Denormals::Any)
    }
}

/// Why a program did not get as far as running.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stop {
    /// The lifter refused it.
    Refused(String),
    /// It lifted and a later stage refused it (validation, a pass, the WGSL that a Direct3D 9 program runs as).
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

/// Lift a program of any bytecode; a Direct3D 9 program that reads pipeline state at Direct3D 9's defaults, with
/// what makes it again for other state.
pub fn lift(bytes: &[u8]) -> Result<(Lifted, Option<d3d9::Stateful>), Stop> {
    let bytecode = Bytecode::of(bytes);
    if bytecode == Bytecode::D3d9 {
        return d3d9::lift(bytes);
    }
    let program = match bytecode {
        Bytecode::Dxbc => iridia_dxbc::lift(bytes),
        _ => iridia_dxil::lift(bytes),
    }
    .map_err(|e| Stop::Refused(e.to_string()))?;
    let wgsl = iridia::emit::wgsl(&program)
        .map_err(|e| format!("WGSL: {e}"))
        .and_then(|text| read_back(&text));
    Ok((
        Lifted {
            bytecode,
            program: Arc::new(program),
            ir: None,
            wgsl,
        },
        None,
    ))
}

/// A DXIL program lifted onto the IR ([`iridia_dxil::ir`]) and structured, as `iridia-exec` runs it: what an
/// amplification program and the mesh program it launches run as, which the naga-path lift does not take.
pub fn lift_ir(bytes: &[u8]) -> Result<iridia_ir::Program, Stop> {
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
    /// Direct3D 9: its IR as `iridia-exec` runs it ([`ProgramShader::from_ir`]), or why not; and with denormals
    /// flushed.
    pub ir_shader: Option<Result<ProgramShader, String>>,
    /// Direct3D 9: [`Ready::ir_shader`] with 32-bit denormals flushed, where the program leaves them open.
    pub ir_flushed: Option<Result<ProgramShader, String>>,
    /// It as `iridia-exec` runs any program, or why not.
    pub executable: Result<Executable, String>,
}

/// The form of a program a draw runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// As lifted (DXBC, DXIL), or its WGSL read back (Direct3D 9).
    Lifted,
    /// Its IR (Direct3D 9).
    Ir,
}

impl Ready {
    /// A lifted program made runnable in every form it has.
    fn of(lifted: Lifted) -> Self {
        let shader = ProgramShader::new((*lifted.program).clone()).map_err(|e| e.to_string());
        let flushed = lifted.denormals_open().then(|| {
            let mut program = (*lifted.program).clone();
            program.modes.flush_denormals = true;
            ProgramShader::new(program).map_err(|e| e.to_string())
        });
        let executable = Executable::new(&lifted.program).map_err(|e| e.to_string());
        // the IR in the register view the WGSL runs in
        let ir = lifted.ir.as_ref().map(|p| {
            let mut ir = p.ir.clone();
            d3d9::own_registers(&mut ir.interface).map(|()| ir)
        });
        let ir_shader = ir
            .clone()
            .map(|p| ProgramShader::from_ir(p?).map_err(|e| e.to_string()));
        let ir_flushed = ir.filter(|_| lifted.denormals_open()).map(|p| {
            let mut p = p?;
            p.float.f32 = iridia_ir::Denormals::Flush;
            ProgramShader::from_ir(p).map_err(|e| e.to_string())
        });
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
    /// program leaves them open; `None` where the program has no such form (an IR of DXBC or DXIL).
    pub fn shader(&self, form: Form, flushed: bool) -> Option<&Result<ProgramShader, String>> {
        let (kept, flush) = match form {
            Form::Lifted => (Some(&self.shader), self.flushed.as_ref()),
            Form::Ir => (self.ir_shader.as_ref(), self.ir_flushed.as_ref()),
        };
        if flushed { flush.or(kept) } else { kept }
    }
}

/// The stop of a lock another thread panicked holding.
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
    /// The programs `bytes` finds by name.
    pub fn new(bytes: impl Fn(&str) -> Result<Vec<u8>, String> + Send + Sync + 'f) -> Self {
        Programs {
            bytes: Box::new(bytes),
            lifted: Mutex::new(BTreeMap::new()),
            stateful: Mutex::new(BTreeMap::new()),
            specialised: Mutex::new(BTreeMap::new()),
        }
    }

    /// Program `name`'s bytes.
    pub fn bytes(&self, name: &str) -> Result<Vec<u8>, String> {
        (self.bytes)(name)
    }

    /// Program `name`, lifted; at Direct3D 9's default state where it reads any.
    pub fn get(&self, name: &str) -> Result<Arc<Ready>, Stop> {
        if let Some(done) = self.lifted.lock().map_err(poisoned)?.get(name) {
            return done.clone();
        }
        let bytes = (self.bytes)(name).map_err(Stop::Failed)?;
        let done = lift(&bytes).map(|(lifted, stateful)| {
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
