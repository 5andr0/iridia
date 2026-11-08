//! The writer's state, and the small pieces every operation is written with: lines of the function body, temporaries, the
//! text of a value's name, and a lane-by-lane call of a helper function.

use alloc::{
    collections::{BTreeMap, BTreeSet},
    format,
    string::{String, ToString},
    vec::Vec,
};

use iridia_ir::interface::{Resource, ResourceId, ResourceKind, Value};
use iridia_ir::target::{Binding, BindingPlan, EntryIo};
use iridia_ir::{Function, Program, Res, Scalar, Type, ValueId, VarId};

use crate::Error;
use crate::flow::Construct;
use crate::helpers::Helper;
use crate::memory::Atomics;
use crate::names::Names;
use crate::origin::Sites;
use crate::profile::Profile;
use crate::text::{self, err};

/// A result that may be an error of the writer.
pub(crate) type R<T = String> = Result<T, Error>;

/// One member of the entry point's result structure, and what it is made of.
#[derive(Clone, Debug)]
pub(crate) struct Member {
    /// The member's name.
    pub name: String,
    /// Its location or built-in: the program's own, or the one its consumer reads it at.
    pub binding: Binding,
    /// What the program's output is: a pixel target narrower than four lanes is carried as four.
    pub carried: Value,
    /// What the member's type is: the carried value, or what a consumer reads it as.
    pub linked: Value,
    /// The interface outputs it is made of: one, or every clip distance.
    pub slots: Vec<usize>,
    /// Whether it is the clip distances array.
    pub clip: bool,
}

/// The writer of one program.
pub(crate) struct Writer<'a> {
    pub program: &'a Program,
    pub f: &'a Function,
    pub plan: &'a BindingPlan,
    pub profile: &'a Profile,
    pub names: Names,
    pub io: EntryIo,
    /// The result structure's members, in order.
    pub members: Vec<Member>,
    /// The name of each resource's global.
    pub globals: Vec<(ResourceId, String)>,
    /// The name of the hidden counter's global of each view that has one.
    pub counters: Vec<(ResourceId, String)>,
    /// The name of each variable of the function.
    pub vars: Vec<String>,
    pub atomics: Atomics,
    pub helpers: BTreeSet<Helper>,
    /// The function body so far.
    pub out: String,
    pub depth: usize,
    pub next: u32,
    /// Whether each value has been defined yet.
    pub defined: Vec<bool>,
    /// The values defined inside a region that has ended.
    pub hidden: BTreeSet<usize>,
    pub constructs: Vec<Construct>,
    /// Per vector value built of known scalars, the scalar of each lane (names of values already bound, or literals): a
    /// lane read takes the scalar, so that a lane that is the same in every invocation stays so for a compiler's
    /// uniformity analysis, which takes a whole vector as the least uniform of its lanes.
    pub lanes: BTreeMap<u32, Vec<String>>,
    /// The scalar of each lane of each value an `If` arm yielded (set where the arm ends in a yield, taken by the `If`).
    pub gathered: Option<Vec<Vec<String>>>,
    /// Whether each value is read where the body places it: by an operation a node places, by a node or by a region's end.
    pub read: Vec<bool>,
    /// Where each lane of each value comes from, for the lanes a variable need not hold.
    pub sites: Sites<'a>,
    /// Per storage (a construct's result or a loop's parameter), the scalar each lane is in place of a variable: a lane the
    /// construct leaves the same as a value bound before it.
    pub forward: BTreeMap<String, Vec<Option<String>>>,
    /// Per variable that holds a vector, the variable of each lane.
    pub split: BTreeMap<String, Vec<String>>,
    /// The local of each output slot that a member carries.
    pub outputs: Vec<Option<String>>,
    /// Structure declarations the module needs besides the result's.
    pub decls: Vec<String>,
    /// The function's ray queries as the text declares them.
    pub queries: Vec<crate::rayquery::QueryVars>,
}

impl Writer<'_> {
    // lines of the body

    pub(crate) fn line(&mut self, text: &str) {
        for _ in 0..self.depth {
            self.out.push_str("    ");
        }
        self.out.push_str(text);
        self.out.push('\n');
    }

    /// The text `f` writes at one level deeper, not yet in the body.
    pub(crate) fn capture(&mut self, f: impl FnOnce(&mut Self) -> R<()>) -> R<String> {
        let outer = core::mem::take(&mut self.out);
        let before = self.defined.clone();
        self.depth = self.depth.saturating_add(1);
        let done = f(self);
        self.depth = self.depth.saturating_sub(1);
        let inner = core::mem::replace(&mut self.out, outer);
        // what the text declares is out of scope after it (a region's braces close)
        for (i, (now, was)) in self.defined.iter().zip(&before).enumerate() {
            if *now && !*was {
                self.hidden.insert(i);
            }
        }
        done?;
        Ok(inner)
    }

    /// Whether the name of `v` can be read here: it has been defined, and not inside a region that has ended.
    pub(crate) fn visible(&self, v: ValueId) -> bool {
        self.defined.get(v.0 as usize).copied().unwrap_or(false) && !self.hidden.contains(&(v.0 as usize))
    }

    /// A temporary made of `expr`, named for use more than once: `let x3 = expr;`.
    pub(crate) fn tmp(&mut self, expr: &str) -> String {
        let name = format!("x{}", self.next);
        self.next = self.next.saturating_add(1);
        self.line(&format!("let {name} = {expr};"));
        name
    }

    /// A fresh `var` of type `ty` with the initial value `init` (zero where `None`): `var x3: T = init;`.
    pub(crate) fn var(&mut self, ty: &str, init: Option<&str>) -> String {
        let name = format!("x{}", self.next);
        self.next = self.next.saturating_add(1);
        match init {
            Some(i) => self.line(&format!("var {name}: {ty} = {i};")),
            None => self.line(&format!("var {name}: {ty};")),
        }
        name
    }

    // values

    pub(crate) fn name(&self, v: ValueId) -> String {
        format!("v{}", v.0)
    }

    /// The name of a value that has been defined.
    pub(crate) fn get(&self, v: ValueId) -> R<String> {
        if self.defined.get(v.0 as usize).copied().unwrap_or(false) {
            Ok(self.name(v))
        } else {
            Err(err(format!("%{} is used before it is computed", v.0)))
        }
    }

    pub(crate) fn vty(&self, v: ValueId) -> R<Type> {
        self.f.ty(v).ok_or_else(|| err(format!("%{} has no type", v.0)))
    }

    /// Define value `v` as `expr`: `let v3 = expr;`.
    pub(crate) fn define(&mut self, v: ValueId, expr: &str) -> R<()> {
        let name = self.name(v);
        self.line(&format!("let {name} = {expr};"));
        self.mark(v)
    }

    /// The scalar of each lane of `v`: those known, or `v`'s own lanes.
    pub(crate) fn lane_names(&self, v: ValueId) -> R<Vec<String>> {
        if let Some(known) = self.lanes.get(&v.0) {
            return Ok(known.clone());
        }
        let x = self.get(v)?;
        let t = self.vty(v)?;
        if t.lanes == 1 {
            return Ok(alloc::vec![x]);
        }
        (0..t.lanes).map(|k| self.lane(&x, t, k)).collect()
    }

    /// Record that `v` has its name defined.
    pub(crate) fn mark(&mut self, v: ValueId) -> R<()> {
        *self
            .defined
            .get_mut(v.0 as usize)
            .ok_or_else(|| err(format!("%{} is out of range", v.0)))? = true;
        Ok(())
    }

    // resources and variables

    pub(crate) fn resource(&self, id: ResourceId) -> R<&Resource> {
        self.program
            .interface
            .resources
            .iter()
            .find(|r| r.id == id)
            .ok_or_else(|| err(format!("resource {} is not in the interface", id.0)))
    }

    pub(crate) fn kind(&self, id: ResourceId) -> R<&ResourceKind> {
        Ok(&self.resource(id)?.kind)
    }

    /// The global of a resource.
    pub(crate) fn global(&self, id: ResourceId) -> R<&str> {
        self.globals
            .iter()
            .find(|(r, _)| *r == id)
            .map(|(_, n)| n.as_str())
            .ok_or_else(|| err(format!("resource {} has no binding", id.0)))
    }

    /// The global of resource `r.id` as an operation names it: the global, or for an array of resources (a binding array)
    /// the element `r.index` picks, element 0 where it names none. An element past the array is undefined, as in Direct3D:
    /// the index is not clamped (an unbounded array's length is the bound data's, not the program's).
    pub(crate) fn pick(&self, r: Res) -> R<String> {
        let g = self.global(r.id)?;
        if !self.resource(r.id)?.is_array() {
            return Ok(g.to_string());
        }
        Ok(match r.index {
            None => format!("{g}[0u]"),
            Some(i) => format!("{g}[{}]", self.index_u32(i)?),
        })
    }

    /// The array of a buffer an operation names (its words, elements or constant rows): its global, or for an element of an
    /// array of buffers the element's one member `data` (a binding array of buffers is one of structures).
    pub(crate) fn buffer(&self, r: Res) -> R<String> {
        let picked = self.pick(r)?;
        Ok(if self.resource(r.id)?.is_array() {
            format!("{picked}.data")
        } else {
            picked
        })
    }

    pub(crate) fn var_name(&self, var: VarId) -> R<&str> {
        self.vars
            .get(var.0 as usize)
            .map(String::as_str)
            .ok_or_else(|| err(format!("variable {} does not exist", var.0)))
    }

    // typed text

    /// `t`'s name.
    pub(crate) fn ty(&self, t: Type) -> R<String> {
        text::ty(t)
    }

    /// The same bits as `to`'s scalar type: `bitcast<vec3<u32>>(x)`; `x` itself where the scalar is the same.
    pub(crate) fn bitcast(&self, x: &str, from: Type, to: Scalar) -> R<String> {
        if from.scalar == to {
            Ok(x.to_string())
        } else {
            Ok(format!("bitcast<{}>({x})", text::ty(from.with(to))?))
        }
    }

    /// A literal `bits` of `s`, in each lane of `like`.
    pub(crate) fn konst(&self, like: Type, s: Scalar, bits: u64) -> R<String> {
        text::splat(like, s, bits)
    }

    /// An `f32` literal in each lane of `like`.
    pub(crate) fn f32s(&self, like: Type, x: f32) -> R<String> {
        self.konst(like, Scalar::F32, u64::from(x.to_bits()))
    }

    /// A `u32` literal in each lane of `like`.
    pub(crate) fn u32s(&self, like: Type, x: u32) -> R<String> {
        self.konst(like, Scalar::U32, u64::from(x))
    }

    /// `name` called on each lane: `name(a)` for a scalar, `vec3<R>(name(a.x), name(a.y), name(a.z))` for a vector, each
    /// argument a name of `t`'s lanes.
    pub(crate) fn lanewise(&mut self, helper: Helper, args: &[&str], t: Type, result: Scalar) -> R<String> {
        self.helpers.insert(helper);
        let name = helper.name();
        if t.lanes == 1 {
            return Ok(format!("{name}({})", args.join(", ")));
        }
        let mut parts = Vec::new();
        for k in 0..t.lanes {
            let c = text::lane(k)?;
            let lane_args: Vec<String> = args.iter().map(|a| format!("{a}.{c}")).collect();
            parts.push(format!("{name}({})", lane_args.join(", ")));
        }
        Ok(format!("{}({})", text::ty(t.with(result))?, parts.join(", ")))
    }

    /// Lane `k` of `x` of type `t`: `x.y`, or `x` for a scalar.
    pub(crate) fn lane(&self, x: &str, t: Type, k: u8) -> R<String> {
        if t.lanes == 1 {
            Ok(x.to_string())
        } else {
            Ok(format!("{x}.{}", text::lane(k)?))
        }
    }

    /// A value of type `t` built a lane at a time: `f` gives lane `k` as a scalar.
    pub(crate) fn per_lane(&mut self, t: Type, mut f: impl FnMut(&mut Self, u8) -> R<String>) -> R<String> {
        let parts = (0..t.lanes).map(|k| f(self, k)).collect::<Result<Vec<_>, _>>()?;
        Ok(match parts.as_slice() {
            [one] => one.clone(),
            _ => format!("{}({})", text::ty(t)?, parts.join(", ")),
        })
    }

    /// An index (`u32` or `i32`) as a `u32` of the same bits.
    pub(crate) fn index_u32(&self, v: ValueId) -> R<String> {
        let x = self.get(v)?;
        match self.vty(v)? {
            Type::U32 => Ok(x),
            Type::I32 => Ok(format!("bitcast<u32>({x})")),
            t => Err(err(format!("an index of {t}"))),
        }
    }
}

/// A bool vector's lanes all true, as one bool (the scalar itself for one lane).
pub(crate) fn all_of(x: &str, lanes: u8) -> String {
    if lanes == 1 {
        x.to_string()
    } else {
        format!("all({x})")
    }
}
