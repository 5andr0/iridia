//! The subset of Direct3D 9 programs this crate lifts, over the records `d3d9-reader` reads: `vs_3_0` and `ps_3_0`
//! straight-line float arithmetic, texture sampling and discard. Whatever the reader reads that is outside it (another
//! version, flow control, predication, the loop counter, integer and boolean constants, an opcode or register this
//! crate does not lift) is refused here by name and word offset.
use std::collections::BTreeMap;

use d3d9_reader::{Comment, Declared, Item, Literal, Modifier, RegisterType, TextureType};
use iridia::{Stage, TextureDim};

use crate::{ConstantTable, Result, error};

/// A float register in the supported stage-specific banks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Register {
    /// Temporary register `rN`, N below 32.
    Temporary(u32),
    /// Input register `vN`: below 16 in a vertex program, below 10 in a pixel program.
    Input(u32),
    /// Float constant register `cN`: below 256 in a vertex program, below 224 in a pixel program.
    Constant(u32),
    /// Sampler `sN`: below 4 in a vertex program, below 16 in a pixel program.
    Sampler(u32),
    /// Colour output `oCN` of a pixel program, N below 4.
    Target(u32),
    /// A declared vertex output register, distinct from pixel render targets.
    Output(u32),
    /// The vertex address register `a0`, written by `mova` and read as the index of a relative constant.
    Address,
}

/// A source token after decoding its swizzle and modifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Source {
    /// The register read.
    pub register: Register,
    /// The four source lanes (0 x, 1 y, 2 z, 3 w) that supply the operand's x, y, z and w, from the two-bit fields of the
    /// token at bits 16, 18, 20 and 22.
    pub swizzle: [u8; 4],
    /// True for the `abs` modifier (the token's source modifier 11, or 12 with negation).
    pub absolute: bool,
    /// True for the negate modifier (source modifier 1, or 12 with the absolute value taken first).
    pub negate: bool,
    /// For a float constant `c[a0.l + N]` of a vertex program, the lane `l` (0 x to 3 w) of the address register `a0`
    /// that is added to the register number; `None` for a plain register.
    pub relative: Option<u8>,
}

impl Source {
    /// Source register lanes needed for selected result lanes after the operand's swizzle.
    pub(crate) fn read_mask(self, needed: u8) -> u8 {
        self.swizzle
            .iter()
            .enumerate()
            .filter(|(n, _)| needed & (1 << n) != 0)
            .fold(0, |mask, (_, lane)| mask | (1 << lane))
    }
}

/// A destination token; masks keep their original lane positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Destination {
    /// The register written.
    pub register: Register,
    /// The lanes written, bit 0 x to bit 3 w (bits 16 to 19 of the token); never empty.
    pub mask: u8,
    /// True for the `_sat` modifier (clamp the result to 0..1).
    pub saturate: bool,
    /// A request that hardware may execute at partial precision. The lifter uses f32 and reports it.
    pub partial_precision: bool,
}

/// The operations this lifter reads. Declarations and immediate definitions are kept separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Opcode {
    /// `nop` (opcode 0).
    Nop,
    /// `mov` (1): copy.
    Mov,
    /// `add` (2).
    Add,
    /// `sub` (3).
    Sub,
    /// `mad` (4): multiply and add.
    Mad,
    /// `mul` (5).
    Mul,
    /// `rcp` (6): reciprocal of the x lane.
    Rcp,
    /// `rsq` (7): reciprocal square root of the x lane.
    Rsq,
    /// `dp3` (8): three-lane dot product.
    Dp3,
    /// `dp4` (9): four-lane dot product.
    Dp4,
    /// `min` (10).
    Min,
    /// `max` (11).
    Max,
    /// `slt` (12): 1 where the first operand is less than the second, else 0.
    Slt,
    /// `sge` (13): 1 where the first operand is at least the second, else 0.
    Sge,
    /// `exp` (14): 2 raised to the x lane.
    Exp,
    /// `log` (15): base 2 logarithm of the x lane.
    Log,
    /// `lrp` (18): linear interpolation.
    Lrp,
    /// `frc` (19): fractional part.
    Frc,
    /// `pow` (32): first operand's x lane raised to the second's.
    Pow,
    /// `abs` (35).
    Abs,
    /// `nrm` (36): normalise the three-lane vector.
    Nrm,
    /// `sincos` (37): the cosine of the x lane in x and its sine in y (the Shader Model 3 form, one source).
    SinCos,
    /// `mova` (46): round the operand to the nearest integer into the vertex address register.
    Mova,
    /// `texkill` (65): discard the pixel when any lane of the register is negative.
    TexKill,
    /// `texld` (66): sample a texture; `Instruction::control` says plain, projected or biased.
    Tex,
    /// `cmp` (88): pick the second operand where the first is at least zero, the third elsewhere.
    Cmp,
    /// `dp2add` (90): two-lane dot product plus the x lane of the third operand.
    Dp2Add,
}

impl Opcode {
    fn decode(op: u16, at: usize) -> Result<Self> {
        use Opcode::*;
        Ok(match op {
            0 => Nop,
            1 => Mov,
            2 => Add,
            3 => Sub,
            4 => Mad,
            5 => Mul,
            6 => Rcp,
            7 => Rsq,
            8 => Dp3,
            9 => Dp4,
            10 => Min,
            11 => Max,
            12 => Slt,
            13 => Sge,
            14 => Exp,
            15 => Log,
            18 => Lrp,
            19 => Frc,
            32 => Pow,
            35 => Abs,
            36 => Nrm,
            37 => SinCos,
            46 => Mova,
            65 => TexKill,
            66 => Tex,
            88 => Cmp,
            90 => Dp2Add,
            _ => {
                let name = match op {
                    25 => "CALL",
                    26 => "CALLNZ",
                    27 => "LOOP",
                    28 => "RET",
                    29 => "ENDLOOP",
                    30 => "LABEL",
                    33 => "CRS",
                    34 => "SGN",
                    38 => "REP",
                    39 => "ENDREP",
                    40 => "IF",
                    41 => "IFC",
                    42 => "ELSE",
                    43 => "ENDIF",
                    44 => "BREAK",
                    45 => "BREAKC",
                    47 => "DEFB",
                    48 => "DEFI",
                    91 => "DSX",
                    92 => "DSY",
                    93 => "TEXLDD",
                    94 => "SETP",
                    95 => "TEXLDL",
                    96 => "BREAKP",
                    _ => "unrecognized",
                };
                return Err(error(
                    at,
                    format!("opcode {op} {name} outside the supported SM3 subset"),
                ));
            }
        })
    }
}

/// An executable operation. `control` is TEX's projection/bias mode (0, 1, 2); zero for arithmetic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Instruction {
    /// Offset of the instruction token in 32-bit words from the version token, the position a refusal and
    /// `Projection::after_word` name.
    pub word: usize,
    /// The operation.
    pub opcode: Opcode,
    /// The register written (for `texkill`, the register tested, which may be an input); `None` for `nop`.
    pub destination: Option<Destination>,
    /// The source operands in the order the token stream writes them.
    pub sources: Vec<Source>,
    /// For `texld`, the bits 16 to 17 of the instruction token: 0 plain, 1 projected (`texldp`), 2 biased (`texldb`); 0
    /// for every other operation.
    pub control: u8,
}

impl Instruction {
    /// Source operand `n`, an error naming the instruction when it has fewer.
    pub(crate) fn source(&self, n: usize) -> Result<Source> {
        self.sources.get(n).copied().ok_or_else(|| {
            error(
                self.word,
                format!(
                    "{:?} reads source {n}, where it has {}",
                    self.opcode,
                    self.sources.len()
                ),
            )
        })
    }

    /// Arithmetic operand lanes before swizzle; shared by execution and backwards output slicing.
    pub(crate) fn source_mask(&self, operand: usize) -> u8 {
        let mask = self.destination.map_or(0, |d| d.mask);
        match self.opcode {
            Opcode::Rcp | Opcode::Rsq | Opcode::Exp | Opcode::Log | Opcode::Pow | Opcode::SinCos => 1,
            Opcode::Dp3 => 7,
            Opcode::Dp4 => 15,
            Opcode::Nrm => 7 | mask,
            Opcode::Dp2Add if operand == 2 => 1,
            Opcode::Dp2Add => 3,
            _ => mask,
        }
    }
}

/// A declared pixel or vertex input (a `dcl` of an input register).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Input {
    /// The input register number N of `vN`.
    pub register: u32,
    /// The semantic usage (D3DDECLUSAGE: 0 position, 1 blend weight, 2 blend indices, 3 normal, 5 texcoord, 10 color and
    /// so on), bits 0 to 3 of the declaration word; 0 to 13.
    pub usage: u8,
    /// The semantic index (the 1 of `TEXCOORD1`), bits 16 to 19 of the declaration word.
    pub index: u8,
    /// The lanes declared, bit 0 x to bit 3 w.
    pub mask: u8,
    /// True when the declaration carries the centroid modifier.
    pub centroid: bool,
    /// True when the declaration carries the partial-precision modifier.
    pub partial_precision: bool,
}

/// A declared vertex output; disjoint masks can pack distinct semantics into one register.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Output {
    /// The output register number N of `oN`, below 12.
    pub register: u32,
    /// The semantic usage (D3DDECLUSAGE, 0 to 13 except 4, point size), as for `Input::usage`.
    pub usage: u8,
    /// The semantic index.
    pub index: u8,
    /// The lanes of the register this semantic occupies, bit 0 x to bit 3 w.
    pub mask: u8,
}

/// Reflection and executable tokens of one complete `ps_3_0` or `vs_3_0` stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shader {
    /// Vertex or pixel, from the version token (`0xfffe0300` or `0xffff0300`).
    pub stage: Stage,
    /// The declared vertex outputs in declaration order; empty for a pixel program.
    pub outputs: Vec<Output>,
    /// The declared inputs in declaration order.
    pub inputs: Vec<Input>,
    /// The declared samplers by register number, with the texture dimension each declares (2D, cube or volume).
    pub samplers: BTreeMap<u32, TextureDim>,
    /// `def` constants, as exact f32 bit patterns.
    pub definitions: BTreeMap<u32, [u32; 4]>,
    /// The executable instructions in program order.
    pub instructions: Vec<Instruction>,
    /// Every comment, in order, including the one that holds the constant table.
    pub comments: Vec<Comment>,
    /// The CTAB comment decoded; `None` when the program has none.
    pub constants: Option<ConstantTable>,
}

/// The instruction token at `word`, quoted by a refusal.
fn token(bytes: &[u8], word: usize) -> u32 {
    word.checked_mul(4)
        .and_then(|at| bytes.get(at..at.checked_add(4)?))
        .and_then(|b| <[u8; 4]>::try_from(b).ok())
        .map_or(0, u32::from_le_bytes)
}

/// The register a reader register is in the supported banks of `stage`.
fn register(r: d3d9_reader::Register, at: usize, stage: Stage) -> Result<Register> {
    let n = r.number;
    let vertex = stage == Stage::Vertex;
    Ok(match r.kind {
        RegisterType::Temp if n < 32 => Register::Temporary(n),
        RegisterType::Input if n < if vertex { 16 } else { 10 } => Register::Input(n),
        RegisterType::Const if n < if vertex { 256 } else { 224 } => Register::Constant(n),
        RegisterType::Address if vertex => Register::Address,
        RegisterType::Output if vertex && n < 12 => Register::Output(n),
        RegisterType::ColorOut if !vertex && n < 4 => Register::Target(n),
        RegisterType::Sampler if n < if vertex { 4 } else { 16 } => Register::Sampler(n),
        kind => {
            return Err(error(
                at,
                format!(
                    "register type {}, index {n} outside the supported {stage:?} SM3 banks",
                    kind.code()
                ),
            ));
        }
    })
}

/// A destination of the subset: no relative address, a mask, and only the modifiers `declaration` allows
/// (partial precision and centroid on a `dcl`, saturate and partial precision elsewhere).
fn destination(
    d: &d3d9_reader::Destination,
    bytes: &[u8],
    declaration: bool,
    stage: Stage,
) -> Result<Destination> {
    if d.relative.is_some() {
        return Err(error(
            d.word,
            "relative addressing of a destination (not supported)",
        ));
    }
    let register = register(d.register, d.word, stage)?;
    let allowed = if declaration { !d.saturate } else { !d.centroid };
    if d.mask == 0 || d.shift != 0 || !allowed {
        return Err(error(
            d.word,
            format!(
                "destination mask/shift/modifier {:#010x}, where the mask is not empty and the shift is 0",
                token(bytes, d.word)
            ),
        ));
    }
    Ok(Destination {
        register,
        mask: d.mask,
        saturate: d.saturate,
        partial_precision: d.partial_precision,
    })
}

fn source(s: &d3d9_reader::Source, stage: Stage) -> Result<Source> {
    let register = register(s.register, s.word, stage)?;
    let relative = match s.relative {
        // the relative flag is for a float constant of a vertex program only
        Some(_) if !(stage == Stage::Vertex && matches!(register, Register::Constant(_))) => {
            return Err(error(
                s.word,
                "parameter marker/reserved bits or relative addressing (not supported)",
            ));
        }
        Some(r) => {
            let at = r.word.unwrap_or(s.word);
            if r.register.kind != RegisterType::Address {
                return Err(error(
                    at,
                    "relative address token other than the vertex address register a0 (a loop counter aL is not supported)",
                ));
            }
            let [x, y, z, w] = r.swizzle;
            if !(x == y && y == z && z == w) {
                return Err(error(
                    at,
                    "relative address token with a modifier or a swizzle that does not replicate one lane",
                ));
            }
            Some(x)
        }
        None => None,
    };
    let (absolute, negate) = match s.modifier {
        Modifier::None => (false, false),
        Modifier::Negate => (false, true),
        Modifier::Abs => (true, false),
        Modifier::AbsNegate => (true, true),
        other => {
            return Err(error(
                s.word,
                format!("source modifier {} outside NONE/NEG/ABS/ABSNEG", other.code()),
            ));
        }
    };
    Ok(Source {
        register,
        swizzle: s.swizzle,
        absolute,
        negate,
        relative,
    })
}

/// Read a complete raw pixel or vertex shader, including comment/CTAB payloads and END, and check that it is in the
/// subset this crate lifts: the version, the bank limits, declaration order, duplicate definitions, the modifiers and
/// every operation. The reader's own refusals keep their word and text.
pub fn read(bytes: &[u8]) -> Result<Shader> {
    let mut records = d3d9_reader::records(bytes).map_err(|e| error(e.word, e.detail))?;
    // the reader reads the version token first; any but the two this crate lifts is refused before the rest is read
    let stage = match records.version().token() {
        0xffff_0300 => Stage::Pixel,
        0xfffe_0300 => Stage::Vertex,
        _ => {
            return Err(error(
                0,
                format!("version {}: only ps_3_0 and vs_3_0 are lifted", records.version()),
            ));
        }
    };
    let mut shader = Shader {
        stage,
        outputs: Vec::new(),
        inputs: Vec::new(),
        samplers: BTreeMap::new(),
        definitions: BTreeMap::new(),
        instructions: Vec::new(),
        comments: Vec::new(),
        constants: None,
    };
    let mut executable = false;
    while let Some(item) = records.next() {
        // the first record this crate cannot use is refused before the reader reads further
        let item = item.map_err(|e| error(e.word, e.detail))?;
        let word = item.word();
        let instruction = token(bytes, word);
        match item {
            Item::Comment(comment) => {
                if comment.data.first() == Some(&u32::from_le_bytes(*b"CTAB"))
                    && let Some(table) = records.constants()
                {
                    for c in &table.constants {
                        let vertex = stage == Stage::Vertex;
                        let limit = match c.register_set {
                            0 | 1 => 16,
                            3 if vertex => 4,
                            3 => 16,
                            _ if vertex => 256,
                            _ => 224,
                        };
                        if u32::from(c.register).saturating_add(u32::from(c.count)) > limit {
                            return Err(error(
                                c.word,
                                format!(
                                    "CTAB register range c{}+{} is empty or out-of-bank (the bank has {limit} registers)",
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
                if instruction & 0xf0ff_0000 != 0 {
                    return Err(error(
                        word,
                        format!("instruction controls/predication/coissue {instruction:#010x}"),
                    ));
                }
                if executable {
                    return Err(error(word, "DCL after executable code"));
                }
                let d = destination(&dcl.destination, bytes, true, stage)?;
                let at = dcl.destination.word;
                if stage == Stage::Vertex && (d.partial_precision || dcl.destination.centroid) {
                    return Err(error(at, "vertex DCL partial precision or centroid modifier"));
                }
                let semantic = |declared: Declared| match declared {
                    Declared::Semantic { usage, index } => Ok((usage, index)),
                    Declared::Sampler(_) => Err(error(
                        word,
                        "DCL of a texture type for a register that is not a sampler",
                    )),
                };
                match d.register {
                    Register::Output(register) => {
                        let (usage, index) = semantic(dcl.declared)?;
                        if usage == 4 {
                            return Err(error(
                                word.saturating_add(1),
                                "vertex output usage outside the subset (0 to 13, PSIZE not supported)",
                            ));
                        }
                        if usage == 0 && index == 0 && d.mask != 15 {
                            return Err(error(at, "vertex POSITION0 must declare all four lanes"));
                        }
                        if shader.outputs.iter().any(|o| {
                            (o.register == register && o.mask & d.mask != 0)
                                || (o.usage == usage && o.index == index)
                        }) {
                            return Err(error(word, "overlapping output masks or duplicate semantic"));
                        }
                        shader.outputs.push(Output {
                            register,
                            usage,
                            index,
                            mask: d.mask,
                        });
                    }
                    Register::Input(register) => {
                        let (usage, index) = semantic(dcl.declared)?;
                        if shader
                            .inputs
                            .iter()
                            .any(|i| i.register == register || (i.usage == usage && i.index == index))
                        {
                            return Err(error(word, "duplicate input register or semantic"));
                        }
                        shader.inputs.push(Input {
                            register,
                            usage,
                            index,
                            mask: d.mask,
                            centroid: dcl.destination.centroid,
                            partial_precision: d.partial_precision,
                        });
                    }
                    Register::Sampler(n) => {
                        let Declared::Sampler(texture) = dcl.declared else {
                            return Err(error(word, "sampler DCL without a texture type"));
                        };
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
                    _ => {
                        return Err(error(
                            at,
                            "DCL destination is neither input, vertex output nor sampler",
                        ));
                    }
                }
            }
            Item::Definition(def) => {
                if instruction & 0xf0ff_0000 != 0 {
                    return Err(error(
                        word,
                        format!("instruction controls/predication/coissue {instruction:#010x}"),
                    ));
                }
                let Literal::Float(values) = def.literal else {
                    let (op, name) = match def.literal {
                        Literal::Integer(_) => (48, "DEFI"),
                        _ => (47, "DEFB"),
                    };
                    return Err(error(
                        word,
                        format!("opcode {op} {name} outside the supported SM3 subset"),
                    ));
                };
                if executable {
                    return Err(error(word, "DEF after executable code"));
                }
                let d = destination(&def.destination, bytes, false, stage)?;
                let Register::Constant(n) = d.register else {
                    return Err(error(def.destination.word, "DEF destination is not cN"));
                };
                if d.mask != 15
                    || d.saturate
                    || d.partial_precision
                    || shader.definitions.insert(n, values).is_some()
                {
                    return Err(error(word, "duplicate or modified DEF"));
                }
            }
            Item::Instruction(i) => {
                let op = i.opcode.code();
                if i.predicate.is_some()
                    || i.coissue
                    || (i.control != 0 && i.opcode != d3d9_reader::Opcode::Tex)
                {
                    return Err(error(
                        word,
                        format!("instruction controls/predication/coissue {instruction:#010x}"),
                    ));
                }
                let opcode = Opcode::decode(op, word)?;
                // an opcode of the other stage (cmp, dp2add, texld and texkill in a vertex program; slt, sge and mova
                // in a pixel one) is not an instruction of the version, which the reader refuses
                let destination = match &i.destination {
                    Some(d) => Some(destination(d, bytes, false, stage)?),
                    None => None,
                };
                if let (Some(d), Some(raw)) = (destination, &i.destination) {
                    let at = raw.word;
                    let allowed = if opcode == Opcode::Mova {
                        d.register == Register::Address
                    } else {
                        matches!(
                            d.register,
                            Register::Temporary(_) | Register::Target(_) | Register::Output(_)
                        ) || (opcode == Opcode::TexKill && matches!(d.register, Register::Input(_)))
                    };
                    if !allowed {
                        return Err(error(at, format!("{opcode:?} destination {:?}", d.register)));
                    }
                    if let Register::Output(n) = d.register {
                        let declared = shader
                            .outputs
                            .iter()
                            .filter(|o| o.register == n)
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
                    if opcode == Opcode::TexKill && (d.saturate || d.partial_precision) {
                        return Err(error(at, "modified TEXKILL destination"));
                    }
                }
                let sources = i
                    .sources
                    .iter()
                    .map(|s| source(s, stage))
                    .collect::<Result<Vec<_>>>()?;
                shader.instructions.push(Instruction {
                    word,
                    opcode,
                    destination,
                    sources,
                    control: i.control,
                });
                executable = true;
            }
        }
    }
    Ok(shader)
}
