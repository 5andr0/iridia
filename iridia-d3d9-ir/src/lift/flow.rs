//! Flow control and predication: `if`, `if_comp` and `if p0` with `else` and `endif`; `rep` and `loop` with `aL`,
//! `break`, `break_comp` and `breakp`; subroutines (`label`, `call`, `callnz`, `ret`), each call inlined in a block
//! a `ret` leaves; `setp` and the per-lane predicate of a predicated write.

use d3d9_reader::{Comparison, Modifier};
use iridia_ir::{Binary, Cmp as Compare, End, Op, Origin, Scalar, Type, Unary, Unit, ValueId};

use super::{Frame, Lift};
use crate::{Instruction, Opcode, Register, Result, Source, error};

/// The IR comparison of a D3D9 one.
pub(super) fn cmp(c: Comparison) -> Compare {
    match c {
        Comparison::Greater => Compare::Gt,
        Comparison::Equal => Compare::Eq,
        Comparison::GreaterEqual => Compare::Ge,
        Comparison::Less => Compare::Lt,
        Comparison::NotEqual => Compare::Ne,
        Comparison::LessEqual => Compare::Le,
    }
}

impl Lift<'_> {
    /// Split the instructions at the first `label`: the main program before it, and each subroutine from its
    /// `label` to the next. The main program's last `ret` is its end.
    pub(super) fn subroutines(&mut self) -> Result<std::ops::Range<usize>> {
        let all = &self.source.instructions;
        let labels: Vec<usize> = (0..all.len())
            .filter(|&n| all.get(n).is_some_and(|i| i.opcode == Opcode::Label))
            .collect();
        let mut main_end = labels.first().copied().unwrap_or(all.len());
        for (k, &at) in labels.iter().enumerate() {
            let i = all.get(at).ok_or_else(|| error(0, "a label past the program"))?;
            let Register::Label(l) = i.source(0)?.register else {
                return Err(error(i.word, "label of a register that is not a label"));
            };
            let end = labels.get(k.saturating_add(1)).copied().unwrap_or(all.len());
            if self.subroutines.insert(l, at.saturating_add(1)..end).is_some() {
                return Err(error(i.word, format!("label l{l} is defined twice")));
            }
        }
        if main_end > 0
            && all
                .get(main_end.saturating_sub(1))
                .is_some_and(|i| i.opcode == Opcode::Ret)
        {
            main_end = main_end.saturating_sub(1);
        }
        Ok(0..main_end)
    }

    /// Lift the instructions of `range` in order.
    pub(super) fn run(&mut self, range: std::ops::Range<usize>) -> Result<()> {
        let source = self.source;
        let end = range.end;
        for n in range {
            let i = source
                .instructions
                .get(n)
                .ok_or_else(|| error(0, "an instruction past the program"))?;
            self.at = i.word;
            self.b.at(Some(Origin {
                unit: Unit::Word,
                offset: u32::try_from(i.word).unwrap_or(u32::MAX),
            }));
            // a pixel 1.x instruction reads what its co-issued partner read: the writes before it commit now
            if !i.coissue {
                self.flush()?;
            }
            if i.opcode == Opcode::Ret && n.saturating_add(1) == end && self.calls.last().is_some() {
                // a subroutine's last `ret` is its end
                continue;
            }
            if !self.flow_instruction(i)? {
                self.instruction(i)?;
            }
        }
        Ok(())
    }

    /// The bool of a predicate source: `p0`'s lanes after its swizzle (four, or the first one alone for `if`,
    /// `callnz` and `breakp`), inverted by `!`.
    fn predicate_lanes(&mut self, p: Source, lanes: u8) -> Result<ValueId> {
        if p.register != Register::Predicate {
            return Err(error(
                self.at,
                format!("{} read as a predicate", p.register.name()),
            ));
        }
        let &(var, written) = self
            .registers
            .get(&Register::Predicate)
            .ok_or_else(|| error(self.at, "p0 is read before setp writes it"))?;
        let needed = p.read_mask(if lanes == 1 { 1 } else { 15 });
        if needed & !written != 0 {
            return Err(error(
                self.at,
                format!(
                    "p0: lanes {:#x} are read before setp writes them",
                    needed & !written
                ),
            ));
        }
        let all = self.b.value(Type::vector(Scalar::Bool, 4), Op::LoadVar(var));
        let mut v = if lanes == 1 {
            self.lane(all, p.swizzle[0])?
        } else if p.swizzle != [0, 1, 2, 3] {
            self.swizzle(all, &p.swizzle)?
        } else {
            all
        };
        match p.modifier {
            Modifier::None => {}
            Modifier::Not => v = self.unary(Unary::Not, v)?,
            other => {
                return Err(error(self.at, format!("predicate modifier {}", other.code())));
            }
        }
        Ok(v)
    }

    /// The four-lane predicate of a predicated write.
    pub(super) fn predicate(&mut self, p: Source) -> Result<ValueId> {
        self.predicate_lanes(p, 4)
    }

    /// The condition of `if`, `callnz` (source `n`): a boolean constant or a predicate lane.
    fn condition(&mut self, s: Source) -> Result<ValueId> {
        match s.register {
            Register::Boolean(n) => {
                let v = self.boolean(n)?;
                if s.modifier == Modifier::Not {
                    return self.unary(Unary::Not, v);
                }
                Ok(v)
            }
            Register::Predicate => self.predicate_lanes(s, 1),
            other => Err(error(
                self.at,
                format!(
                    "{} as a condition, where it takes a boolean or the predicate",
                    other.name()
                ),
            )),
        }
    }

    /// The comparison of the first lanes of `if_comp`'s and `break_comp`'s sources.
    fn comparison(&mut self, i: &Instruction) -> Result<ValueId> {
        let c = i
            .comparison()
            .ok_or_else(|| error(self.at, format!("{} has no comparison", i.opcode.name())))?;
        let a = self.source(i.source(0)?, 1)?;
        let a = self.lane(a, 0)?;
        let b = self.source(i.source(1)?, 1)?;
        let b = self.lane(b, 0)?;
        self.binary(Binary::FCmp(cmp(c)), a, b)
    }

    /// The label of the loop a `break` leaves.
    fn innermost_loop(&self) -> Result<iridia_ir::LabelId> {
        for f in self.flow.iter().rev() {
            match f {
                Frame::Loop { label, .. } => return Ok(*label),
                Frame::Call { .. } => break,
                Frame::If { .. } => {}
            }
        }
        Err(error(self.at, "break outside a loop"))
    }

    /// Lift `i` if it is flow control or `setp`; false for every other instruction.
    fn flow_instruction(&mut self, i: &Instruction) -> Result<bool> {
        use Opcode::*;
        if i.predicate.is_some()
            && matches!(
                i.opcode,
                If | IfC
                    | Else
                    | EndIf
                    | Rep
                    | EndRep
                    | Loop
                    | EndLoop
                    | Break
                    | BreakC
                    | BreakP
                    | Call
                    | CallNz
                    | Ret
                    | Label
            )
        {
            return Err(error(self.at, format!("a predicated {}", i.opcode.name())));
        }
        match i.opcode {
            If | IfC => {
                let cond = if i.opcode == If {
                    self.condition(i.source(0)?)?
                } else {
                    self.comparison(i)?
                };
                self.b.open_if(cond);
                self.flow.push(Frame::If { otherwise: false });
            }
            Else => match self.flow.last_mut() {
                Some(Frame::If { otherwise }) if !*otherwise => {
                    *otherwise = true;
                    self.b.else_(End::Yield(vec![])).map_err(|e| error(self.at, e))?;
                }
                _ => return Err(error(self.at, "else without its if")),
            },
            EndIf => match self.flow.pop() {
                Some(Frame::If { .. }) => {
                    self.b
                        .close(End::Yield(vec![]), &[])
                        .map_err(|e| error(self.at, e))?;
                }
                _ => return Err(error(self.at, "endif without its if")),
            },
            Rep | Loop => {
                let counter_source = i.source(if i.opcode == Loop { 1 } else { 0 })?;
                let Register::Integer(n) = counter_source.register else {
                    return Err(error(
                        self.at,
                        format!(
                            "{} counts with {}, where it takes iN",
                            i.opcode.name(),
                            counter_source.register.name()
                        ),
                    ));
                };
                if i.opcode == Loop && i.source(0)?.register != Register::Loop {
                    return Err(error(self.at, "loop counts in a register that is not aL"));
                }
                let constant = self.integer(n)?;
                let constant = if counter_source.swizzle == [0, 1, 2, 3] {
                    constant
                } else {
                    self.swizzle(constant, &counter_source.swizzle)?
                };
                // the iteration count is unsigned, decremented at the top so that a step of zero still ends
                let count = self.lane(constant, 0)?;
                let count = self.b.value(Type::U32, Op::Bitcast(count));
                let iterations = self.b.var(&format!("count{}", self.flow.len()), Type::U32);
                self.b.store_var(iterations, count);
                let counter = if i.opcode == Loop {
                    let start = self.lane(constant, 1)?;
                    let step = self.lane(constant, 2)?;
                    let al = self.b.var(&format!("aL{}", self.flow.len()), Type::I32);
                    self.b.store_var(al, start);
                    Some((al, step))
                } else {
                    None
                };
                let (label, _) = self.b.open_loop(&[]).map_err(|e| error(self.at, e))?;
                let left = self.b.value(Type::U32, Op::LoadVar(iterations));
                let zero = self.b.u32(0);
                let done = self.binary(Binary::UCmp(Compare::Eq), left, zero)?;
                self.b.open_if(done);
                self.b
                    .close(End::Br(label, vec![]), &[])
                    .map_err(|e| error(self.at, e))?;
                let one = self.b.u32(1);
                let next = self.binary(Binary::ISub, left, one)?;
                self.b.store_var(iterations, next);
                self.flow.push(Frame::Loop { label, counter });
            }
            EndRep | EndLoop => {
                let Some(Frame::Loop { label, counter }) = self.flow.pop() else {
                    return Err(error(self.at, format!("{} without its loop", i.opcode.name())));
                };
                if (i.opcode == EndLoop) != counter.is_some() {
                    return Err(error(
                        self.at,
                        format!("{} ends the other kind of loop", i.opcode.name()),
                    ));
                }
                if let Some((al, step)) = counter {
                    let value = self.b.value(Type::I32, Op::LoadVar(al));
                    let next = self.binary(Binary::IAdd, value, step)?;
                    self.b.store_var(al, next);
                }
                self.b
                    .close(End::Continue(label, vec![]), &[])
                    .map_err(|e| error(self.at, e))?;
            }
            Break | BreakC | BreakP => {
                let label = self.innermost_loop()?;
                let cond = match i.opcode {
                    Break => None,
                    BreakC => Some(self.comparison(i)?),
                    _ => Some(self.condition(i.source(0)?)?),
                };
                match cond {
                    Some(c) => {
                        self.b.open_if(c);
                        self.b
                            .close(End::Br(label, vec![]), &[])
                            .map_err(|e| error(self.at, e))?;
                    }
                    // what follows an unconditional break is not reached; it goes in an arm never taken
                    None => {
                        let always = self.b.bool(true);
                        self.b.open_if(always);
                        self.b
                            .close(End::Br(label, vec![]), &[])
                            .map_err(|e| error(self.at, e))?;
                    }
                }
            }
            Call | CallNz => {
                let Register::Label(l) = i.source(0)?.register else {
                    return Err(error(self.at, "call of a register that is not a label"));
                };
                if self.calls.contains(&l) {
                    return Err(error(self.at, format!("l{l} calls itself")));
                }
                let range = self
                    .subroutines
                    .get(&l)
                    .cloned()
                    .ok_or_else(|| error(self.at, format!("call of l{l}, which is not defined")))?;
                let conditional = i.opcode == CallNz;
                if conditional {
                    let cond = self.condition(i.source(1)?)?;
                    self.b.open_if(cond);
                }
                let label = self.b.open_block();
                self.flow.push(Frame::Call { label });
                self.calls.push(l);
                let at = self.at;
                self.run(range)?;
                self.flush()?;
                self.at = at;
                self.calls.pop();
                match self.flow.pop() {
                    Some(Frame::Call { .. }) => {}
                    _ => return Err(error(self.at, format!("l{l} leaves flow control open"))),
                }
                self.b
                    .close(End::Yield(vec![]), &[])
                    .map_err(|e| error(self.at, e))?;
                if conditional {
                    self.b
                        .close(End::Yield(vec![]), &[])
                        .map_err(|e| error(self.at, e))?;
                }
            }
            Ret => {
                let label = self
                    .flow
                    .iter()
                    .rev()
                    .find_map(|f| match f {
                        Frame::Call { label } => Some(*label),
                        _ => None,
                    })
                    .ok_or_else(|| error(self.at, "ret inside flow control of the main program"))?;
                let always = self.b.bool(true);
                self.b.open_if(always);
                self.b
                    .close(End::Br(label, vec![]), &[])
                    .map_err(|e| error(self.at, e))?;
            }
            Label => return Err(error(self.at, "a label reached by falling through")),
            SetP => {
                let d = i.destination()?;
                let c = i
                    .comparison()
                    .ok_or_else(|| error(self.at, "setp has no comparison"))?;
                let a = self.source(i.source(0)?, d.mask)?;
                let b = self.source(i.source(1)?, d.mask)?;
                let v = self.binary(Binary::FCmp(cmp(c)), a, b)?;
                self.write(i, d, v)?;
            }
            _ => return Ok(false),
        }
        Ok(true)
    }
}
