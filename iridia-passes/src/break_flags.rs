//! Route nonlocal exits through typed result payloads. Ordinary SSA results and loop parameters stay in place;
//! only constructs crossed by an escaping exit carry its tag and arguments. A guard carries its enclosing
//! region's output tuple, never the whole function's variable environment.

use alloc::{string::String, vec::Vec};
use iridia_ir::{Body, End, Function, LabelId, Node, Region};

mod router;

/// A construct being walked: what a region's own exits leave, and what a route may arrive at.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Frame {
    If,
    Block(LabelId),
    Loop(LabelId),
    Switch(LabelId),
}

impl Frame {
    fn label(self) -> Option<LabelId> {
        match self {
            Frame::If => None,
            Frame::Block(l) | Frame::Loop(l) | Frame::Switch(l) => Some(l),
        }
    }
    /// Whether a target's `break` leaves it.
    fn breakable(self) -> bool {
        matches!(self, Frame::Loop(_) | Frame::Switch(_))
    }
    fn of(n: &Node) -> Option<Frame> {
        match n {
            Node::If { .. } => Some(Frame::If),
            Node::Block { label, .. } => Some(Frame::Block(*label)),
            Node::Loop { label, .. } => Some(Frame::Loop(*label)),
            Node::Switch { label, .. } => Some(Frame::Switch(*label)),
            _ => None,
        }
    }
}

/// Whether an exit to the frame at `at` of `stack` is taken directly: a break of the innermost loop or switch, a
/// continue of the innermost loop.
fn direct(stack: &[Frame], at: usize, resume: bool) -> bool {
    let inner = stack.get(at.saturating_add(1)..).unwrap_or_default();
    match (stack.get(at), resume) {
        (Some(Frame::Loop(_)), true) => !inner.iter().any(|f| matches!(f, Frame::Loop(_))),
        (Some(f), false) if f.breakable() => !inner.iter().any(|f| f.breakable()),
        _ => false,
    }
}

/// Whether any exit in `r` (within `stack`) needs routing.
fn needs(r: &Region, stack: &mut Vec<Frame>) -> bool {
    let inner = r.nodes.iter().any(|n| match Frame::of(n) {
        Some(frame) => {
            stack.push(frame);
            let x = n.regions().into_iter().any(|inner| needs(inner, stack));
            stack.pop();
            x
        }
        None => false,
    });
    inner
        || match &r.end {
            Some(e @ (End::Br(t, _) | End::Continue(t, _))) => {
                let resume = matches!(e, End::Continue(..));
                match stack.iter().rposition(|f| f.label() == Some(*t)) {
                    Some(at) => {
                        let between = stack.get(at.saturating_add(1)..).unwrap_or_default();
                        // a block's exit from the end of its own body is its fall-off
                        let tail =
                            matches!(stack.get(at), Some(Frame::Block(_))) && between.is_empty() && !resume;
                        !direct(stack, at, resume) && !tail
                    }
                    None => false,
                }
            }
            _ => false,
        }
}

/// Each `Br` to a block from the end of the block's own body made its fall-off.
fn tails(r: &mut Region, stack: &mut Vec<Frame>) {
    for n in &mut r.nodes {
        if let Some(frame) = Frame::of(n) {
            stack.push(frame);
            for inner in n.regions_mut() {
                tails(inner, stack);
            }
            stack.pop();
        }
    }
    if let Some(End::Br(t, vs)) = &mut r.end
        && stack.last() == Some(&Frame::Block(*t))
    {
        r.end = Some(End::Yield(core::mem::take(vs)));
    }
}

/// Route the exits of a `Regions` function that innermost-only `break` and `continue` cannot take. Returns the
/// number of routed exits. A graph body is refused. With no route, only a block's own tail exits become yields.
/// Existing values and loop parameters remain SSA; route payloads use the existing construct result form.
pub fn break_flags(f: &mut Function) -> Result<usize, String> {
    let Body::Regions(mut body) = core::mem::replace(&mut f.body, Body::Regions(Region::default())) else {
        return Err("break_flags takes a Regions body".into());
    };
    if !needs(&body, &mut Vec::new()) {
        tails(&mut body, &mut Vec::new());
        f.body = Body::Regions(body);
        return Ok(0);
    }
    crate::simplify::prune_unused(f, &mut body);
    let (body, count) = router::route(f, body)?;
    f.body = Body::Regions(body);
    crate::simplify(f);
    Ok(count)
}
