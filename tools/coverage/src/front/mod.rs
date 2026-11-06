//! The bytecode front ends, as iridia's public pipeline is today: the one place a suite runner learns how a program is
//! read, lifted, written as WGSL and made runnable, so the runners follow the pipeline when it changes.
//!
//! - Direct3D 9 token streams: `iridia-d3d9-ir` lifts into iridia's IR, `iridia-passes` promotes and simplifies it,
//!   `iridia-naga` writes WGSL, and the WGSL read back by naga's front end is what `iridia-exec` runs; the IR itself
//!   runs in `iridia-exec-ir` beside it ([`d3d9`]).
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

/// Lift a program of any bytecode.
pub fn lift(bytes: &[u8]) -> Result<Lifted, Stop> {
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
    Ok(Lifted {
        bytecode,
        program: Arc::new(program),
        ir: None,
        wgsl,
    })
}

/// A program lifted and made runnable.
#[derive(Debug)]
pub struct Ready {
    /// How far it got.
    pub lifted: Lifted,
    /// It as `iridia-exec` runs a vertex or pixel program, or why not.
    pub shader: Result<ProgramShader, String>,
    /// It as `iridia-exec` runs any program, or why not.
    pub executable: Result<Executable, String>,
}

/// Each program lifted so far, by name: it, ready to run, or why it stopped.
pub type Lifts = BTreeMap<String, Result<Arc<Ready>, Stop>>;

type Source<'f> = Box<dyn Fn(&str) -> Result<Vec<u8>, String> + Send + Sync + 'f>;

/// A suite's programs by name, lifted once and shared by the runs that use them.
pub struct Programs<'f> {
    bytes: Source<'f>,
    lifted: Mutex<Lifts>,
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
        }
    }

    /// Program `name`'s bytes.
    pub fn bytes(&self, name: &str) -> Result<Vec<u8>, String> {
        (self.bytes)(name)
    }

    /// Program `name`, lifted.
    pub fn get(&self, name: &str) -> Result<Arc<Ready>, Stop> {
        let poisoned = |_| Stop::Failed("a poisoned lock".into());
        if let Some(done) = self.lifted.lock().map_err(poisoned)?.get(name) {
            return done.clone();
        }
        let bytes = (self.bytes)(name).map_err(Stop::Failed)?;
        let done = lift(&bytes).map(|lifted| {
            let shader = ProgramShader::new((*lifted.program).clone()).map_err(|e| e.to_string());
            let executable = Executable::new(&lifted.program).map_err(|e| e.to_string());
            Arc::new(Ready {
                lifted,
                shader,
                executable,
            })
        });
        self.lifted
            .lock()
            .map_err(poisoned)?
            .insert(name.to_owned(), done.clone());
        done
    }

    /// Every program lifted so far.
    pub fn lifted(&self) -> Lifts {
        self.lifted.lock().map(|l| l.clone()).unwrap_or_default()
    }
}
