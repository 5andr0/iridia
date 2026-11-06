//! A program as this crate lifts it, over the records `d3d9-reader` reads: the version, its registers by bank, the
//! declared and implied inputs and outputs, the samplers, the literal definitions and the instructions in order.
//!
//! Every register is checked against its version's banks here, and the inputs and outputs a version implies by
//! register rather than declares are made explicit: a pixel 1.x program's `vN` (`COLOR`) and `tN` (`TEXCOORD`) and
//! its colour in `r0`; a vertex 1.x or 2.x program's `oPos`, `oDN`, `oTN`, `oFog` and `oPts`. What the reader reads
//! that this crate does not lift is refused by name and word where the lift meets it.
use std::collections::BTreeMap;

use d3d9_reader::{Comment, Declared, Item, Literal, Modifier, RegisterType, TextureType};
use iridia_ir::interface::{Stage, TextureDim};

pub use d3d9_reader::{Comparison, Opcode, Version};

use crate::{ConstantTable, Result, error};

/// A register, by bank. The numbers are the register's own; which exist depends on the version.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Register {
    /// Temporary `rN`.
    Temporary(u32),
    /// Input `vN`: a vertex attribute, or a pixel program's colour input before 3.0 and its declared input in 3.0.
    Input(u32),
    /// A pixel program's texture register `tN` before 3.0: a texture coordinate input, and in pixel 1.1 to 1.3 the
    /// register a texture operation writes.
    Texture(u32),
    /// Float constant `cN`.
    Constant(u32),
    /// Integer constant `iN`.
    Integer(u32),
    /// Boolean constant `bN`.
    Boolean(u32),
    /// Sampler `sN`.
    Sampler(u32),
    /// The vertex address register `a0`.
    Address,
    /// The loop counter `aL`.
    Loop,
    /// The predicate `p0`.
    Predicate,
    /// A pixel program's colour output `oCN`.
    Target(u32),
    /// A pixel program's depth output `oDepth`.
    Depth,
    /// A vertex 3.0 program's output `oN`, with the semantic its declaration gives.
    Output(u32),
    /// A vertex program's position `oPos` before 3.0.
    Position,
    /// A vertex program's fog `oFog` before 3.0 (the x lane).
    Fog,
    /// A vertex program's point size `oPts` before 3.0 (the x lane).
    PointSize,
    /// A vertex program's colour output `oDN` before 3.0.
    Color(u32),
    /// A vertex program's texture coordinate output `oTN` before 3.0.
    TexCoord(u32),
    /// A label `lN` of a subroutine.
    Label(u32),
}

impl Register {
    /// The register's name as Microsoft's disassembler writes it.
    pub fn name(self) -> String {
        match self {
            Register::Temporary(n) => format!("r{n}"),
            Register::Input(n) => format!("v{n}"),
            Register::Texture(n) => format!("t{n}"),
            Register::Constant(n) => format!("c{n}"),
            Register::Integer(n) => format!("i{n}"),
            Register::Boolean(n) => format!("b{n}"),
            Register::Sampler(n) => format!("s{n}"),
            Register::Address => "a0".into(),
            Register::Loop => "aL".into(),
            Register::Predicate => "p0".into(),
            Register::Target(n) => format!("oC{n}"),
            Register::Depth => "oDepth".into(),
            Register::Output(n) => format!("o{n}"),
            Register::Position => "oPos".into(),
            Register::Fog => "oFog".into(),
            Register::PointSize => "oPts".into(),
            Register::Color(n) => format!("oD{n}"),
            Register::TexCoord(n) => format!("oT{n}"),
            Register::Label(n) => format!("l{n}"),
        }
    }
}

/// The index register of a relatively addressed source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Index {
    /// A lane (0 x to 3 w) of the address register `a0`; always x in a vertex 1.x program.
    Address(u8),
    /// The loop counter `aL`.
    Loop,
}

/// A source operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Source {
    /// The register read.
    pub register: Register,
    /// The register lane (0 x to 3 w) each of the operand's x, y, z and w reads.
    pub swizzle: [u8; 4],
    /// The source modifier, as stored.
    pub modifier: Modifier,
    /// The index register of `c[a0.x + N]`, `c[aL + N]`, `v[aL + N]`; `None` for a plain register.
    pub relative: Option<Index>,
}

impl Source {
    /// The register lanes the operand reads for the result lanes `needed`, after its swizzle (and, for `_dz` and
    /// `_dw`, the lane divided by).
    pub fn read_mask(self, needed: u8) -> u8 {
        let read = self
            .swizzle
            .iter()
            .enumerate()
            .filter(|(n, _)| needed & (1 << n) != 0)
            .fold(0, |mask, (_, lane)| mask | (1 << lane));
        match self.modifier {
            Modifier::DivideZ => read | 4,
            Modifier::DivideW => read | 8,
            _ => read,
        }
    }

    /// True for the negating modifiers (`-r`, `-r_bias`, `-r_bx2`, `-r_x2`, `-r_abs`).
    pub fn negate(self) -> bool {
        matches!(
            self.modifier,
            Modifier::Negate
                | Modifier::BiasNegate
                | Modifier::SignNegate
                | Modifier::X2Negate
                | Modifier::AbsNegate
        )
    }

    /// True for `r_abs` and `-r_abs`.
    pub fn absolute(self) -> bool {
        matches!(self.modifier, Modifier::Abs | Modifier::AbsNegate)
    }
}

/// A destination operand; the mask keeps its lane positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Destination {
    /// The register written.
    pub register: Register,
    /// The lanes written, bit 0 x to bit 3 w.
    pub mask: u8,
    /// `_sat`.
    pub saturate: bool,
    /// `_pp`: a request that hardware may compute at partial precision.
    pub partial_precision: bool,
    /// The pixel 1.x result shift: the result is multiplied by 2 to this power (`_x2` 1 to `_d8` -3) before `_sat`.
    pub shift: i8,
}

/// An instruction: everything but declarations, definitions and comments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Instruction {
    /// The word of the instruction token, counted from the version token.
    pub word: usize,
    /// The opcode.
    pub opcode: Opcode,
    /// The opcode's controls: for `tex` from pixel 1.4 on 0 `texld`, 1 `texldp`, 2 `texldb`; for `if_comp`,
    /// `break_comp` and `setp_comp` the comparison; 0 otherwise.
    pub control: u8,
    /// The pixel 1.x co-issue flag (`+`): the instruction runs beside the one before it, reading what that one read.
    pub coissue: bool,
    /// The predicate it runs under: `p0` with its swizzle, and [`Modifier::Not`] for `!p0`.
    pub predicate: Option<Source>,
    /// The destination; `None` for an opcode that has none.
    pub destination: Option<Destination>,
    /// The sources, in stream order.
    pub sources: Vec<Source>,
}

impl Instruction {
    /// Source `n`, a refusal naming the instruction when it has fewer.
    pub fn source(&self, n: usize) -> Result<Source> {
        self.sources.get(n).copied().ok_or_else(|| {
            error(
                self.word,
                format!(
                    "{} reads source {n}, where it has {}",
                    self.opcode.name(),
                    self.sources.len()
                ),
            )
        })
    }

    /// The destination, a refusal when it has none.
    pub fn destination(&self) -> Result<Destination> {
        self.destination
            .ok_or_else(|| error(self.word, format!("{} has no destination", self.opcode.name())))
    }

    /// The comparison of `if_comp`, `break_comp` and `setp_comp`.
    pub fn comparison(&self) -> Option<Comparison> {
        if !matches!(self.opcode, Opcode::IfC | Opcode::BreakC | Opcode::SetP) {
            return None;
        }
        Some(match self.control {
            1 => Comparison::Greater,
            2 => Comparison::Equal,
            3 => Comparison::GreaterEqual,
            4 => Comparison::Less,
            5 => Comparison::NotEqual,
            _ => Comparison::LessEqual,
        })
    }

    /// The lanes of source `operand` the operation reads, before its swizzle, for the lanes it writes.
    pub fn source_mask(&self, operand: usize) -> u8 {
        use Opcode::*;
        let mask = self.destination.map_or(0, |d| d.mask);
        match self.opcode {
            Rcp | Rsq | Exp | Log | ExpP | LogP | Pow => 1,
            SinCos if operand == 0 => 1,
            // the 2.0 form's constants, which the operation does not need
            SinCos => 0,
            Dp3 | M3x2 | M3x3 | M3x4 => 7,
            Dp4 | M4x3 | M4x4 => 15,
            Nrm => 7 | mask,
            Dp2Add if operand == 2 => 1,
            Dp2Add => 3,
            Crs => 7,
            Lit => 0b1011,
            Dst if operand == 0 => 0b0110,
            Dst => 0b1010,
            // `sgn`'s second and third sources are scratch registers it may overwrite
            Sgn if operand > 0 => 0,
            Bem => 3,
            _ => mask,
        }
    }

    /// The constant registers a matrix operation reads from its second source on: `m4x4` 4, `m3x2` 2 and so on.
    pub fn matrix_rows(&self) -> Option<u32> {
        match self.opcode {
            Opcode::M4x4 | Opcode::M3x4 => Some(4),
            Opcode::M4x3 | Opcode::M3x3 => Some(3),
            Opcode::M3x2 => Some(2),
            _ => None,
        }
    }
}

/// A declared input, or one a pixel 1.x program implies by reading `vN` or `tN`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Input {
    /// `vN`, or a pixel program's `tN`.
    pub register: Register,
    /// The semantic usage (`D3DDECLUSAGE`: 0 position, 5 texcoord, 10 color and so on).
    pub usage: u8,
    /// The semantic index.
    pub index: u8,
    /// The lanes declared, bit 0 x to bit 3 w.
    pub mask: u8,
    /// `_centroid`.
    pub centroid: bool,
    /// `_pp`.
    pub partial_precision: bool,
}

/// A vertex output: declared in 3.0 (disjoint masks can pack semantics into one register), implied by register before.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Output {
    /// `oN` in 3.0; `oPos`, `oDN`, `oTN`, `oFog` or `oPts` before.
    pub register: Register,
    /// The semantic usage, as for [`Input::usage`].
    pub usage: u8,
    /// The semantic index.
    pub index: u8,
    /// The register lanes this semantic occupies.
    pub mask: u8,
}

/// The `D3DDECLUSAGE` numbers this crate names registers with.
pub mod usage {
    /// `POSITION`.
    pub const POSITION: u8 = 0;
    /// `PSIZE`.
    pub const PSIZE: u8 = 4;
    /// `TEXCOORD`.
    pub const TEXCOORD: u8 = 5;
    /// `COLOR`.
    pub const COLOR: u8 = 10;
    /// `FOG`.
    pub const FOG: u8 = 11;
}

/// One complete program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shader {
    /// The version token's version.
    pub version: Version,
    /// Vertex or pixel.
    pub stage: Stage,
    /// The vertex outputs: declared, in declaration order (3.0), or implied, in register order (before 3.0). Empty for
    /// a pixel program, whose outputs are the colour and depth registers it writes.
    pub outputs: Vec<Output>,
    /// The inputs: declared, in declaration order, or implied (pixel 1.x), colours then texture coordinates.
    pub inputs: Vec<Input>,
    /// The samplers by register, with the texture dimension each declares; a pixel 1.x program's are its texture
    /// stages, whose dimension is pipeline state (`StateParam::TextureDim`) and given here as 2D.
    pub samplers: BTreeMap<u32, TextureDim>,
    /// `def` constants, as f32 bits.
    pub definitions: BTreeMap<u32, [u32; 4]>,
    /// `defi` constants.
    pub integers: BTreeMap<u32, [i32; 4]>,
    /// `defb` constants.
    pub booleans: BTreeMap<u32, bool>,
    /// The instructions in order.
    pub instructions: Vec<Instruction>,
    /// Every comment, including the constant table's.
    pub comments: Vec<Comment>,
    /// The CTAB comment decoded.
    pub constants: Option<ConstantTable>,
}

impl Shader {
    /// True for a pixel 1.x program.
    pub fn pixel_1x(&self) -> bool {
        self.stage == Stage::Pixel && self.version.major == 1
    }

    /// True for a vertex program before 3.0, whose outputs are its registers.
    pub fn fixed_outputs(&self) -> bool {
        self.stage == Stage::Vertex && self.version.major < 3
    }
}

/// How many registers of a bank a version has; `None` for a bank it does not have.
fn limit(kind: RegisterType, v: Version) -> Option<u32> {
    use RegisterType::*;
    let vertex = v.stage == d3d9_reader::Stage::Vertex;
    let (major, minor) = (v.major, v.minor);
    let n = match kind {
        Temp if vertex => {
            if major >= 3 || minor != 0 && major == 2 {
                32
            } else {
                12
            }
        }
        Temp => match (major, minor) {
            (1, 4) => 6,
            (1, _) => 2,
            (2, 0) => 12,
            _ => 32,
        },
        Input if vertex => 16,
        Input => {
            if major >= 3 {
                10
            } else {
                2
            }
        }
        Const if vertex => 256,
        Const => match major {
            1 => 8,
            2 => 32,
            _ => 224,
        },
        Address if vertex => 1,
        Texture if !vertex && major < 3 => match (major, minor) {
            (1, 4) => 6,
            (1, _) => 4,
            _ => 8,
        },
        RasterizerOut if vertex && major < 3 => 3,
        AttributeOut if vertex && major < 3 => 2,
        Output if vertex => {
            if major >= 3 {
                12
            } else {
                8
            }
        }
        ConstInt if major >= 2 => 16,
        ConstBool if major >= 2 => 16,
        ColorOut if !vertex && major >= 2 => 4,
        DepthOut if !vertex && major >= 2 => 1,
        Sampler if vertex && major >= 3 => 4,
        Sampler if !vertex && major >= 2 => 16,
        Loop if major >= 2 && (vertex || major >= 3) => 1,
        Label if major >= 2 => 2048,
        Predicate if major >= 3 || major == 2 && minor != 0 => 1,
        _ => return None,
    };
    Some(n)
}

/// The register a reader register is, in `version`'s banks.
fn register(r: d3d9_reader::Register, at: usize, version: Version) -> Result<Register> {
    let n = r.number;
    if !limit(r.kind, version).is_some_and(|count| n < count) {
        return Err(error(
            at,
            format!(
                "register type {}, index {n} outside the {version} banks",
                r.kind.code()
            ),
        ));
    }
    let vertex = version.stage == d3d9_reader::Stage::Vertex;
    Ok(match r.kind {
        RegisterType::Temp => Register::Temporary(n),
        RegisterType::Input => Register::Input(n),
        RegisterType::Const => Register::Constant(n),
        RegisterType::Address => Register::Address,
        RegisterType::Texture => Register::Texture(n),
        RegisterType::RasterizerOut => match n {
            0 => Register::Position,
            1 => Register::Fog,
            _ => Register::PointSize,
        },
        RegisterType::AttributeOut => Register::Color(n),
        RegisterType::Output if vertex && version.major < 3 => Register::TexCoord(n),
        RegisterType::Output => Register::Output(n),
        RegisterType::ConstInt => Register::Integer(n),
        RegisterType::ConstBool => Register::Boolean(n),
        RegisterType::ColorOut => Register::Target(n),
        RegisterType::DepthOut => Register::Depth,
        RegisterType::Sampler => Register::Sampler(n),
        RegisterType::Loop => Register::Loop,
        RegisterType::Label => Register::Label(n),
        RegisterType::Predicate => Register::Predicate,
        other => {
            return Err(error(
                at,
                format!("register type {} of {version} is not lifted", other.code()),
            ));
        }
    })
}

/// A destination: no relative address, a lane, and only the modifiers `declaration` allows (partial precision and
/// centroid on a `dcl`, saturate, partial precision and the pixel 1.x shift elsewhere).
fn destination(d: &d3d9_reader::Destination, version: Version, declaration: bool) -> Result<Destination> {
    if d.relative.is_some() {
        return Err(error(
            d.word,
            "relative addressing of a destination is not lifted",
        ));
    }
    let register = register(d.register, d.word, version)?;
    let allowed = if declaration { !d.saturate } else { !d.centroid };
    if d.mask == 0 || !allowed {
        return Err(error(
            d.word,
            format!(
                "destination {} with mask {:#x}{}{}, where a destination writes a lane and only a dcl is _centroid",
                register.name(),
                d.mask,
                if d.saturate { " _sat" } else { "" },
                if d.centroid { " _centroid" } else { "" },
            ),
        ));
    }
    Ok(Destination {
        register,
        mask: d.mask,
        saturate: d.saturate,
        partial_precision: d.partial_precision,
        shift: d.shift,
    })
}

/// Whether `version` has the source modifier: from 2.0 negation and absolute value (`!` of a boolean or predicate
/// is checked where it is read); in pixel 1.x negation, `_bias`, `_bx2` and `1 - r`, and from 1.4 `_x2`, `_dz` and
/// `_dw`.
fn modifier_allowed(m: Modifier, version: Version) -> bool {
    use Modifier::*;
    match m {
        None | Negate | Not => true,
        Abs | AbsNegate => version.major >= 2,
        Bias | BiasNegate | Sign | SignNegate | Complement => {
            version.major == 1 && version.stage == d3d9_reader::Stage::Pixel
        }
        X2 | X2Negate | DivideZ | DivideW => {
            version.major == 1 && version.minor == 4 && version.stage == d3d9_reader::Stage::Pixel
        }
    }
}

fn source(s: &d3d9_reader::Source, version: Version) -> Result<Source> {
    let register = register(s.register, s.word, version)?;
    if !modifier_allowed(s.modifier, version) {
        return Err(error(
            s.word,
            format!(
                "source modifier {} ({:?}), which {version} does not have",
                s.modifier.code(),
                s.modifier
            ),
        ));
    }
    let relative = match s.relative {
        None => None,
        Some(r) => {
            let at = r.word.unwrap_or(s.word);
            match (r.register.kind, register) {
                (RegisterType::Address, Register::Constant(_)) => {
                    let [x, y, z, w] = r.swizzle;
                    // compilers write the lane replicated; Microsoft's assembler writes `c0[a0]` as .xyzw, which reads
                    // a0.x
                    if !(x == y && y == z && z == w || r.swizzle == [0, 1, 2, 3]) {
                        return Err(error(
                            at,
                            "relative address token with a modifier or a swizzle that does not replicate one lane",
                        ));
                    }
                    Some(Index::Address(x))
                }
                (RegisterType::Loop, Register::Constant(_) | Register::Input(_)) => Some(Index::Loop),
                _ => {
                    return Err(error(
                        at,
                        format!("relative addressing of {} is not lifted", register.name()),
                    ));
                }
            }
        }
    };
    Ok(Source {
        register,
        swizzle: s.swizzle,
        modifier: s.modifier,
        relative,
    })
}

/// Whether `opcode` may write `r` in this program.
fn writable(opcode: Opcode, r: Register, shader: &Shader) -> bool {
    use Register::*;
    let v = shader.version;
    match opcode {
        Opcode::Mova => r == Address,
        Opcode::SetP => r == Predicate,
        // texkill names the register it tests
        Opcode::TexKill => matches!(r, Temporary(_) | Input(_) | Texture(_)),
        // vs 1.x writes a0 with mov, which it rounds down
        Opcode::Mov if r == Address => shader.stage == Stage::Vertex && v.major == 1,
        _ => match r {
            Temporary(_) | Target(_) | Depth | Output(_) | Position | Fog | PointSize | Color(_)
            | TexCoord(_) => true,
            // pixel 1.1 to 1.3 texture registers are written by texture operations and read as temporaries
            Texture(_) => shader.pixel_1x() && v.minor < 4,
            _ => false,
        },
    }
}

/// Read a complete program and check every register against its version's banks; the reader's own refusals keep
/// their word and text.
pub fn read(bytes: &[u8]) -> Result<Shader> {
    let mut records = d3d9_reader::records(bytes).map_err(|e| error(e.word, e.detail))?;
    let version = records.version();
    if version.minor == 255 {
        return Err(error(
            0,
            format!("{version}: the software versions are not lifted"),
        ));
    }
    if version.stage == d3d9_reader::Stage::Pixel && version.major == 1 && version.minor == 0 {
        return Err(error(0, "ps_1_0: no Direct3D 9 runtime takes it"));
    }
    if version.stage == d3d9_reader::Stage::Vertex && version.major == 1 && version.minor == 0 {
        return Err(error(0, "vs_1_0: no Direct3D 9 runtime takes it"));
    }
    let stage = match version.stage {
        d3d9_reader::Stage::Vertex => Stage::Vertex,
        d3d9_reader::Stage::Pixel => Stage::Pixel,
    };
    let mut shader = Shader {
        version,
        stage,
        outputs: Vec::new(),
        inputs: Vec::new(),
        samplers: BTreeMap::new(),
        definitions: BTreeMap::new(),
        integers: BTreeMap::new(),
        booleans: BTreeMap::new(),
        instructions: Vec::new(),
        comments: Vec::new(),
        constants: None,
    };
    let mut executable = false;
    while let Some(item) = records.next() {
        let item = item.map_err(|e| error(e.word, e.detail))?;
        let word = item.word();
        match item {
            Item::Comment(comment) => {
                if comment.data.first() == Some(&u32::from_le_bytes(*b"CTAB"))
                    && let Some(table) = records.constants()
                {
                    for c in &table.constants {
                        let kind = match c.register_set {
                            0 => RegisterType::ConstBool,
                            1 => RegisterType::ConstInt,
                            3 => RegisterType::Sampler,
                            _ => RegisterType::Const,
                        };
                        // pixel 1.x samples through texture stages, which the table lists as samplers
                        let banked = if kind == RegisterType::Sampler && shader.pixel_1x() {
                            Some(4)
                        } else {
                            None
                        };
                        let bank = banked.or_else(|| limit(kind, version)).unwrap_or(0);
                        if u32::from(c.register).saturating_add(u32::from(c.count)) > bank {
                            return Err(error(
                                c.word,
                                format!(
                                    "CTAB register range {}+{} is out of the bank (the bank has {bank} registers)",
                                    c.register, c.count
                                ),
                            ));
                        }
                    }
                    shader.constants = Some(table.clone());
                }
                shader.comments.push(comment);
            }
            Item::Declaration(dcl) => {
                if executable {
                    return Err(error(word, "DCL after executable code"));
                }
                let d = destination(&dcl.destination, version, true)?;
                let at = dcl.destination.word;
                match (d.register, dcl.declared) {
                    (Register::Output(register), Declared::Semantic { usage, index }) => {
                        if usage == usage::POSITION && index == 0 && d.mask != 15 {
                            return Err(error(at, "a vertex POSITION0 output must declare all four lanes"));
                        }
                        if shader.outputs.iter().any(|o| {
                            o.register == Register::Output(register) && o.mask & d.mask != 0
                                || o.usage == usage && o.index == index
                        }) {
                            return Err(error(word, "overlapping output masks or a duplicate semantic"));
                        }
                        shader.outputs.push(Output {
                            register: Register::Output(register),
                            usage,
                            index,
                            mask: d.mask,
                        });
                    }
                    (Register::Input(_) | Register::Texture(_), Declared::Semantic { usage, index }) => {
                        // a pixel 2.x program names its inputs by register: vN are colours, tN texture coordinates
                        let (usage, index) = match d.register {
                            Register::Input(n) if stage == Stage::Pixel && version.major < 3 => {
                                (usage::COLOR, u8::try_from(n).unwrap_or(u8::MAX))
                            }
                            Register::Texture(n) => (usage::TEXCOORD, u8::try_from(n).unwrap_or(u8::MAX)),
                            _ => (usage, index),
                        };
                        if shader
                            .inputs
                            .iter()
                            .any(|i| i.register == d.register || i.usage == usage && i.index == index)
                        {
                            return Err(error(word, "a duplicate input register or semantic"));
                        }
                        if stage == Stage::Vertex && (d.partial_precision || dcl.destination.centroid) {
                            return Err(error(at, "a vertex input declared _pp or _centroid"));
                        }
                        shader.inputs.push(Input {
                            register: d.register,
                            usage,
                            index,
                            mask: d.mask,
                            centroid: dcl.destination.centroid,
                            partial_precision: d.partial_precision,
                        });
                    }
                    (Register::Sampler(n), Declared::Sampler(texture)) => {
                        let dim = match texture {
                            TextureType::D2 => TextureDim::D2,
                            TextureType::Cube => TextureDim::Cube,
                            TextureType::Volume => TextureDim::D3,
                        };
                        if d.mask != 15
                            || d.partial_precision
                            || dcl.destination.centroid
                            || shader.samplers.insert(n, dim).is_some()
                        {
                            return Err(error(word, "duplicate or modified sampler declaration"));
                        }
                    }
                    (r, _) => return Err(error(at, format!("a dcl of {} is not lifted", r.name()))),
                }
            }
            Item::Definition(def) => {
                if executable {
                    return Err(error(word, "DEF after executable code"));
                }
                let d = destination(&def.destination, version, false)?;
                if d.mask != 15 || d.saturate || d.partial_precision || d.shift != 0 {
                    return Err(error(word, "duplicate or modified DEF"));
                }
                let duplicate = match (d.register, def.literal) {
                    (Register::Constant(n), Literal::Float(values)) => {
                        shader.definitions.insert(n, values).is_some()
                    }
                    (Register::Integer(n), Literal::Integer(values)) => {
                        shader.integers.insert(n, values).is_some()
                    }
                    (Register::Boolean(n), Literal::Bool(value)) => {
                        shader.booleans.insert(n, value != 0).is_some()
                    }
                    (r, _) => return Err(error(word, format!("a definition of {}", r.name()))),
                };
                if duplicate {
                    return Err(error(word, format!("{} is defined twice", d.register.name())));
                }
            }
            Item::Instruction(i) => {
                let controlled = matches!(
                    i.opcode,
                    Opcode::Tex | Opcode::IfC | Opcode::BreakC | Opcode::SetP
                );
                if i.control != 0 && !controlled {
                    return Err(error(
                        word,
                        format!("instruction controls {:#x} on {}", i.control, i.opcode.name()),
                    ));
                }
                let destination = i
                    .destination
                    .as_ref()
                    .map(|d| destination(d, version, false))
                    .transpose()?;
                if let (Some(d), Some(raw)) = (destination, &i.destination) {
                    let at = raw.word;
                    if !writable(i.opcode, d.register, &shader) {
                        return Err(error(
                            at,
                            format!("{} destination {}", i.opcode.name(), d.register.name()),
                        ));
                    }
                    if let Register::Output(n) = d.register {
                        let declared = shader
                            .outputs
                            .iter()
                            .filter(|o| o.register == Register::Output(n))
                            .fold(0, |m, o| m | o.mask);
                        if d.mask & !declared != 0 {
                            return Err(error(
                                at,
                                format!(
                                    "write to undeclared vertex output lanes {:#x} of o{n}, which declares {declared:#x}",
                                    d.mask & !declared
                                ),
                            ));
                        }
                    }
                    if stage == Stage::Vertex && d.partial_precision {
                        return Err(error(at, "vertex partial precision modifier not supported"));
                    }
                    if i.opcode == Opcode::TexKill && (d.saturate || d.partial_precision || d.shift != 0) {
                        return Err(error(at, "modified TEXKILL destination"));
                    }
                }
                let sources = i
                    .sources
                    .iter()
                    .map(|s| source(s, version))
                    .collect::<Result<Vec<_>>>()?;
                let predicate = i.predicate.as_ref().map(|p| source(p, version)).transpose()?;
                shader.instructions.push(Instruction {
                    word,
                    opcode: i.opcode,
                    control: i.control,
                    coissue: i.coissue,
                    predicate,
                    destination,
                    sources,
                });
                executable = true;
            }
        }
    }
    if shader.pixel_1x() {
        imply_pixel_1x(&mut shader);
    }
    if shader.fixed_outputs() {
        imply_vertex_outputs(&mut shader);
    }
    Ok(shader)
}

/// A pixel 1.x program declares nothing: its colour inputs `vN` are the ones it reads; its texture coordinates `tN`
/// are, in 1.4, the ones `texld`, `texcrd` and `texkill` read, and before 1.4 the ones of the texture operations that
/// name `tN` (all but `texreg2ar`, `texreg2gb` and `texreg2rgb`, which sample at another register's value); its
/// texture stages are the ones a texture operation samples.
fn imply_pixel_1x(shader: &mut Shader) {
    use Opcode::*;
    let mut colors = std::collections::BTreeSet::new();
    let mut coordinates = std::collections::BTreeSet::new();
    let mut stages = BTreeMap::new();
    let ps14 = shader.version.minor == 4;
    for i in &shader.instructions {
        for s in &i.sources {
            match s.register {
                Register::Input(n) => {
                    colors.insert(n);
                }
                Register::Texture(n) if ps14 => {
                    coordinates.insert(n);
                }
                _ => {}
            }
        }
        let Some(d) = i.destination else { continue };
        if ps14 {
            if i.opcode == TexKill
                && let Register::Texture(n) = d.register
            {
                coordinates.insert(n);
            }
            if i.opcode == Tex
                && let Register::Temporary(n) = d.register
            {
                stages.insert(n, TextureDim::D2);
            }
            continue;
        }
        let Register::Texture(n) = d.register else {
            continue;
        };
        if !matches!(i.opcode, TexReg2Ar | TexReg2Gb | TexReg2Rgb) {
            coordinates.insert(n);
        }
        if matches!(
            i.opcode,
            Tex | TexBem
                | TexBemL
                | TexReg2Ar
                | TexReg2Gb
                | TexReg2Rgb
                | TexM3x2Tex
                | TexM3x3Tex
                | TexM3x3Spec
                | TexM3x3VSpec
                | TexDp3Tex
        ) {
            stages.insert(n, TextureDim::D2);
        }
    }
    for n in colors {
        shader.inputs.push(Input {
            register: Register::Input(n),
            usage: usage::COLOR,
            index: u8::try_from(n).unwrap_or(u8::MAX),
            mask: 15,
            centroid: false,
            partial_precision: false,
        });
    }
    for n in coordinates {
        shader.inputs.push(Input {
            register: Register::Texture(n),
            usage: usage::TEXCOORD,
            index: u8::try_from(n).unwrap_or(u8::MAX),
            mask: 15,
            centroid: false,
            partial_precision: false,
        });
    }
    shader.samplers = stages;
}

/// A vertex program before 3.0 declares no outputs: they are the output registers it writes, in register order, each
/// with the lanes written (all four of `oPos`, whose unwritten lanes are zero).
fn imply_vertex_outputs(shader: &mut Shader) {
    let mut written: BTreeMap<Register, u8> = BTreeMap::new();
    for i in &shader.instructions {
        if let Some(d) = i.destination
            && matches!(
                d.register,
                Register::Position
                    | Register::Fog
                    | Register::PointSize
                    | Register::Color(_)
                    | Register::TexCoord(_)
            )
        {
            *written.entry(d.register).or_insert(0) |= d.mask;
        }
    }
    for (register, mask) in written {
        let (usage, index, mask) = match register {
            Register::Position => (usage::POSITION, 0, 15),
            Register::Fog => (usage::FOG, 0, 1),
            Register::PointSize => (usage::PSIZE, 0, 1),
            Register::Color(n) => (usage::COLOR, u8::try_from(n).unwrap_or(u8::MAX), mask),
            Register::TexCoord(n) => (usage::TEXCOORD, u8::try_from(n).unwrap_or(u8::MAX), mask),
            _ => continue,
        };
        shader.outputs.push(Output {
            register,
            usage,
            index,
            mask,
        });
    }
}
