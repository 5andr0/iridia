//! Structured control flow from a control flow graph, for lifters whose source has gotos (DXIL's basic blocks, any
//! SPIR-V-like graph): naga has `if`, `loop`, `switch`, `break`, `continue` and `return`, nothing else.
//!
//! [`structure`] is Ramsey's translation of a reducible graph by its dominator tree ("Beyond Relooper", ICFP 2022):
//! a node that several forward edges reach becomes a labelled block its predecessors branch out of, a loop header a
//! labelled loop its back edges continue. An irreducible graph (a back edge whose target does not dominate its source)
//! is refused by name; nothing is duplicated and no flag variable invented for it.
//!
//! [`lower`] writes the result as naga statements. naga's `break` leaves the innermost loop or switch only, so a
//! branch to a label further out is routed: it sets a jump variable and breaks, and each construct it leaves checks the
//! variable after it and carries on outward. That is the one place a variable is introduced, and only where a branch
//! really crosses a construct.

use std::collections::BTreeMap;

use naga::{
    BinaryOperator, Block, Expression, Handle, LocalVariable, Span, Statement, SwitchCase, SwitchValue,
};

pub use iridia_passes::structure::{Cfg, Code, Error, Label, Term, structure};

use super::{Builder, Kind, Width};

/// What the lowering asks of the lifter that owns the nodes' contents.
pub trait Emit {
    /// The error a lifter's emitters report; it is passed through unchanged.
    type Error;
    /// The node's own statements, into the builder's open block.
    fn node(&mut self, b: &mut Builder, node: usize) -> Result<(), Self::Error>;
    /// The copies of the edge `from -> to`.
    fn edge(&mut self, b: &mut Builder, from: usize, to: usize) -> Result<(), Self::Error>;
    /// The bool of the node's two-way branch.
    fn condition(&mut self, b: &mut Builder, node: usize) -> Result<Handle<Expression>, Self::Error>;
    /// The node's multi-way branch: the selector and the value of each target but the last (the default).
    fn selector(
        &mut self,
        b: &mut Builder,
        node: usize,
    ) -> Result<(Handle<Expression>, Vec<SwitchValue>), Self::Error>;
    /// The node leaves the program: its return.
    fn exit(&mut self, b: &mut Builder, node: usize) -> Result<(), Self::Error>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Action {
    Break(Label),
    Continue(Label),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind_ {
    Block(Label),
    Loop(Label),
    Switch,
}

struct Entry {
    kind: Kind_,
    /// The actions routed out through this construct, to be carried on after it.
    pending: Vec<Action>,
}

struct Lowering<'e, E: Emit> {
    emit: &'e mut E,
    jump: Handle<LocalVariable>,
    ids: BTreeMap<Action, u32>,
    stack: Vec<Entry>,
    used: std::collections::BTreeSet<Label>,
}

fn uses(code: &Code, out: &mut std::collections::BTreeSet<Label>) {
    match code {
        Code::Seq(v) => v.iter().for_each(|c| uses(c, out)),
        Code::Block(_, c) | Code::Loop(_, c) => uses(c, out),
        Code::If(_, a, b) => {
            uses(a, out);
            uses(b, out);
        }
        Code::Switch(_, v) => v.iter().for_each(|c| uses(c, out)),
        Code::Br(l) => {
            out.insert(*l);
        }
        Code::Node(_) | Code::Continue(_) | Code::Edge(..) | Code::Exit(_) => {}
    }
}

impl<E: Emit> Lowering<'_, E> {
    fn id(&mut self, action: Action) -> u32 {
        let next = u32::try_from(self.ids.len())
            .unwrap_or(u32::MAX)
            .saturating_add(1);
        *self.ids.entry(action).or_insert(next)
    }

    /// A construct's body, in a block of its own.
    fn block(&mut self, b: &mut Builder, code: &Code) -> Result<Block, E::Error> {
        b.blocks.push(Block::new());
        let result = self.code(b, code);
        let block = b.blocks.pop().expect("the block just pushed");
        result.map(|()| block)
    }

    /// Close the construct on top of the stack, and carry on what was routed through it.
    fn leave(&mut self, b: &mut Builder) -> Result<(), E::Error> {
        let entry = self.stack.pop().expect("an open construct");
        for action in entry.pending {
            let id = self.id(action);
            let jump = b.local_ptr(self.jump);
            let current = b.load(jump, Kind::U, Width::One);
            let want = b.lit_u(id);
            let hit = b.binary(BinaryOperator::Equal, current, want);
            b.blocks.push(Block::new());
            let jump = b.local_ptr(self.jump);
            let zero = b.lit_u(0);
            b.store(jump, zero);
            self.action(b, action);
            let accept = b.blocks.pop().expect("the arm just pushed");
            b.push(Statement::If {
                condition: hit.expr,
                accept,
                reject: Block::new(),
            });
        }
        Ok(())
    }

    // `destination` is a position of the stack, found by `rposition`, and `top` is the last
    #[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
    fn action(&mut self, b: &mut Builder, action: Action) {
        let destination = self
            .stack
            .iter()
            .rposition(|e| match (action, e.kind) {
                (Action::Break(l), Kind_::Block(m)) => l == m,
                (Action::Continue(l), Kind_::Loop(m)) => l == m,
                _ => false,
            })
            .expect("a label's construct encloses its branches");
        let top = self.stack.len() - 1;
        let direct = match action {
            Action::Break(_) => destination == top,
            // `continue` leaves any switches and restarts the innermost loop-like construct
            Action::Continue(_) => self.stack[destination + 1..]
                .iter()
                .all(|e| e.kind == Kind_::Switch),
        };
        if direct {
            b.push(match action {
                Action::Break(_) => Statement::Break,
                Action::Continue(_) => Statement::Continue,
            });
        } else {
            let id = self.id(action);
            let jump = b.local_ptr(self.jump);
            let value = b.lit_u(id);
            b.store(jump, value);
            b.push(Statement::Break);
            let pending = &mut self.stack[top].pending;
            if !pending.contains(&action) {
                pending.push(action);
            }
        }
    }

    fn code(&mut self, b: &mut Builder, code: &Code) -> Result<(), E::Error> {
        match code {
            Code::Node(n) => self.emit.node(b, *n)?,
            Code::Seq(items) => {
                for item in items {
                    self.code(b, item)?;
                }
            }
            Code::Edge(from, to) => self.emit.edge(b, *from, *to)?,
            Code::Exit(n) => self.emit.exit(b, *n)?,
            Code::Br(l) => self.action(b, Action::Break(*l)),
            Code::Continue(l) => self.action(b, Action::Continue(*l)),
            Code::Block(label, body) => {
                if self.used.contains(label) {
                    self.stack.push(Entry {
                        kind: Kind_::Block(*label),
                        pending: Vec::new(),
                    });
                    let mut inner = self.block(b, body)?;
                    inner.push(Statement::Break, Span::UNDEFINED);
                    b.push(Statement::Loop {
                        body: inner,
                        continuing: Block::new(),
                        break_if: None,
                    });
                    self.leave(b)?;
                } else {
                    self.code(b, body)?;
                }
            }
            Code::Loop(label, body) => {
                self.stack.push(Entry {
                    kind: Kind_::Loop(*label),
                    pending: Vec::new(),
                });
                let mut inner = self.block(b, body)?;
                inner.push(Statement::Break, Span::UNDEFINED);
                b.push(Statement::Loop {
                    body: inner,
                    continuing: Block::new(),
                    break_if: None,
                });
                self.leave(b)?;
            }
            Code::If(node, then, otherwise) => {
                let condition = self.emit.condition(b, *node)?;
                let accept = self.block(b, then)?;
                let reject = self.block(b, otherwise)?;
                b.push(Statement::If {
                    condition,
                    accept,
                    reject,
                });
            }
            Code::Switch(node, targets) => {
                let (selector, values) = self.emit.selector(b, *node)?;
                self.stack.push(Entry {
                    kind: Kind_::Switch,
                    pending: Vec::new(),
                });
                let mut cases = Vec::new();
                for (i, target) in targets.iter().enumerate() {
                    let body = self.block(b, target)?;
                    cases.push(SwitchCase {
                        value: values.get(i).copied().unwrap_or(SwitchValue::Default),
                        body,
                        fall_through: false,
                    });
                }
                b.push(Statement::Switch { selector, cases });
                self.leave(b)?;
            }
        }
        Ok(())
    }
}

/// Write `code` into the builder's open block through `emit`. `jump` is a `u32` local (zero at the start) the
/// lowering uses to carry a branch across the constructs it leaves.
pub fn lower<E: Emit>(
    b: &mut Builder,
    code: &Code,
    emit: &mut E,
    jump: Handle<LocalVariable>,
) -> Result<(), E::Error> {
    let mut used = std::collections::BTreeSet::new();
    uses(code, &mut used);
    let mut lowering = Lowering {
        emit,
        jump,
        ids: BTreeMap::new(),
        stack: Vec::new(),
        used,
    };
    lowering.code(b, code)
}
