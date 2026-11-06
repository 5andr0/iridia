//! Functions: a value arena, variables, and a body in one of the two control phases.

use alloc::{string::String, vec::Vec};

use crate::{LabelId, Op, Type, Unit, ValueId, VarId};

/// Where in the source bytes an operation comes from, for refusals, notes and mapping back.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub struct Origin {
    /// What `offset` counts.
    pub unit: Unit,
    /// The offset of the source instruction.
    pub offset: u32,
}

/// The precision an operation is allowed to compute at: a permission, never a statement of what it computes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Precision {
    /// The type's own precision.
    #[default]
    Full,
    /// Any precision of at least 16 bits (D3D9 `_pp`, DXBC `min16float`, GLSL `mediump`).
    AtLeast16,
}

/// A value: its type, the operation that defines it, its source origin and the precision it may be computed at.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ValueDef {
    /// The value's type.
    pub ty: Type,
    /// The operation.
    pub op: Op,
    /// Where it comes from in the source.
    pub origin: Option<Origin>,
    /// The precision it may be computed at.
    pub precision: Precision,
}

/// A mutable variable: a source register until `promote` turns its loads and stores into values.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Var {
    /// Its name, for the text form and target names (`r0`).
    pub name: String,
    /// Its type.
    pub ty: Type,
}

/// A step of a region or a block.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Node {
    /// The value is computed here (its operation's state reads happen here).
    Value(ValueId),
    /// The variable takes the value.
    StoreVar(VarId, ValueId),
    /// Lanes of the interface output `slot` take the value: the lanes of mask `.1` (bit n is lane n, at least one,
    /// within the varying's width), in ascending order, from the value's lanes in order. The value has the varying's
    /// scalar type and one lane per lane written. A lane never written is zero when the program returns.
    StoreOutput(u32, u8, ValueId),
    /// Discards the invocation of a pixel program (D3D's `texkill` and `discard`, WGSL's `discard`) without ending
    /// it: it goes on as a helper invocation, which its quad's derivatives still read, and its later output stores have
    /// no effect. It ends as killed, with the outputs it stored before.
    Demote,
    /// Two-way choice. Each arm ends in a yield of the results' values, or an exit that leaves it.
    If {
        /// A scalar bool.
        cond: ValueId,
        /// The region run where `cond` holds.
        accept: Region,
        /// The region run where it does not.
        reject: Region,
        /// The values defined by the construct (`Op::Param`), given by each arm's yield.
        results: Vec<ValueId>,
    },
    /// A region that `End::Br(label)` leaves early, with values for its results.
    Block {
        /// The label exits name.
        label: LabelId,
        /// The region.
        body: Region,
        /// The values the construct defines (`Op::Param`), given by the body's yield and by each `Br` to it.
        results: Vec<ValueId>,
    },
    /// A loop: `End::Continue(label)` runs the body again with new parameter values; a yield or `Br(label)` leaves it.
    Loop {
        /// The label exits name.
        label: LabelId,
        /// The loop-carried values (`Op::Param`), visible in the body.
        params: Vec<ValueId>,
        /// The parameters' values on entry.
        inits: Vec<ValueId>,
        /// The body.
        body: Region,
        /// The values the construct defines (`Op::Param`), given by the body's yield and each `Br` to it.
        results: Vec<ValueId>,
    },
}

/// How a region ends.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum End {
    /// Falls off the end, giving its construct's results these values.
    Yield(Vec<ValueId>),
    /// Leaves the enclosing `Block` or `Loop` named by the label, with values for its results.
    Br(LabelId, Vec<ValueId>),
    /// Restarts the enclosing `Loop` named by the label, with values for its parameters.
    Continue(LabelId, Vec<ValueId>),
    /// Leaves the function.
    Return,
    /// Discards the invocation and ends it: [`Node::Demote`] then `Return`.
    Kill,
    /// Cannot be reached.
    Unreachable,
}

/// A sequence of nodes and how it ends.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct Region {
    /// The nodes, in order.
    pub nodes: Vec<Node>,
    /// How it ends; `None` only while a builder is filling it.
    pub end: Option<End>,
}

/// A branch target of a `Cfg` block: the block and the values of its parameters.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Target {
    /// The block, by index.
    pub block: u32,
    /// The values its parameters take.
    pub args: Vec<ValueId>,
}

/// How a `Cfg` block ends.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Terminator {
    /// Continues to the target.
    Jump(Target),
    /// Two-way branch on a scalar bool.
    Branch {
        /// The condition.
        cond: ValueId,
        /// Where it holds.
        accept: Target,
        /// Where it does not.
        reject: Target,
    },
    /// Leaves the function.
    Return,
    /// Discards the invocation.
    Kill,
    /// Cannot be reached.
    Unreachable,
}

/// A basic block of the `Cfg` phase: parameters, plain nodes (no constructs) and a terminator.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Block {
    /// Its parameters (`Op::Param`).
    pub params: Vec<ValueId>,
    /// Its nodes: values, stores; never a construct.
    pub nodes: Vec<Node>,
    /// How it ends.
    pub term: Terminator,
}

/// A function body, in one of the two control phases.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Body {
    /// Structured regions: what structured sources build and structured targets take.
    Regions(Region),
    /// A control flow graph; block 0 is the entry.
    Cfg(Vec<Block>),
}

/// A function: its values, variables and body.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Function {
    /// The name (`main` for the entry point).
    pub name: String,
    /// Every value, by `ValueId`.
    pub values: Vec<ValueDef>,
    /// Every variable, by `VarId`.
    pub vars: Vec<Var>,
    /// The body.
    pub body: Body,
}

impl Function {
    /// The definition of a value, if the id is in range.
    pub fn value(&self, v: ValueId) -> Option<&ValueDef> {
        self.values.get(v.0 as usize)
    }
    /// The type of a value, if the id is in range.
    pub fn ty(&self, v: ValueId) -> Option<Type> {
        self.value(v).map(|d| d.ty)
    }
}
