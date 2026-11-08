//! The module's datalayout and the memory laid out by it: the bytes of a task payload, a ray payload and a hit's
//! attributes, which Direct3D lays out as C does (each field at its natural alignment, a bool four bytes), so the
//! programs that share the bytes may declare different structures over them.

use dxil_reader::module::{Module, Type, TypeId};
use iridia::{Leaf, LeafKind};

/// The ABI alignments a module's datalayout states, in bytes, by bit width; LLVM's defaults where it states none.
#[derive(Clone, Debug)]
pub(crate) struct DataLayout {
    integers: Vec<(u32, u32)>,
    floats: Vec<(u32, u32)>,
}

/// The most scalars a structure's description holds, a bound on an array of scalars in a structure.
const MAX_LEAVES: usize = 4096;

/// `n` rounded up to a multiple of `align`.
fn round_up(n: u32, align: u32) -> u32 {
    n.div_ceil(align.max(1)).saturating_mul(align.max(1))
}

impl DataLayout {
    /// The layout of datalayout string `s` (`e-m:e-p:32:32-i1:32-i8:32-i16:32-...`).
    pub(crate) fn parse(s: &str) -> Result<DataLayout, String> {
        // LLVM's defaults
        let mut integers = vec![(1, 1), (8, 1), (16, 2), (32, 4), (64, 4)];
        let mut floats = vec![(16, 2), (32, 4), (64, 8), (128, 16)];
        for spec in s.split('-') {
            let (table, rest) = match spec.as_bytes().first() {
                Some(b'i') => (&mut integers, spec.get(1..).unwrap_or("")),
                Some(b'f') => (&mut floats, spec.get(1..).unwrap_or("")),
                _ => continue,
            };
            let mut parts = rest.split(':');
            let (Some(Ok(bits)), Some(Ok(abi))) = (
                parts.next().map(str::parse::<u32>),
                parts.next().map(str::parse::<u32>),
            ) else {
                return Err(format!("datalayout entry {spec}"));
            };
            if abi == 0 || abi % 8 != 0 {
                return Err(format!("datalayout entry {spec}: an alignment of {abi} bits"));
            }
            match table.iter_mut().find(|(b, _)| *b == bits) {
                Some(entry) => entry.1 = abi / 8,
                None => table.push((bits, abi / 8)),
            }
        }
        Ok(DataLayout { integers, floats })
    }

    /// The datalayout `m` states.
    pub(crate) fn of(m: &Module) -> Result<DataLayout, String> {
        match m.datalayout.as_deref() {
            Some(s) => DataLayout::parse(s),
            None => Err("memory laid out in bytes in a module that states no datalayout".into()),
        }
    }

    fn integer(&self, bits: u32) -> Option<u32> {
        self.integers.iter().find(|(b, _)| *b == bits).map(|&(_, a)| a)
    }

    fn float(&self, bits: u32) -> Option<u32> {
        self.floats.iter().find(|(b, _)| *b == bits).map(|&(_, a)| a)
    }

    /// The ABI alignment and allocation size in bytes of a value of LLVM type `ty` of `m`.
    pub(crate) fn size_align(&self, m: &Module, ty: TypeId) -> Result<(u32, u32), String> {
        let scalar = |align: Option<u32>, bits: u32, what: &str| -> Result<(u32, u32), String> {
            let Some(align) = align else {
                return Err(format!(
                    "memory laid out in bytes with a {what} the datalayout gives no alignment"
                ));
            };
            Ok((align, round_up(bits.div_ceil(8), align)))
        };
        match m.types.get(ty).ok_or_else(|| format!("type {ty}"))? {
            Type::Integer(bits) => scalar(self.integer(*bits), *bits, "integer"),
            Type::Half => scalar(self.float(16), 16, "half"),
            Type::Float => scalar(self.float(32), 32, "float"),
            Type::Double => scalar(self.float(64), 64, "double"),
            Type::Array(n, e) | Type::Vector(n, e) => {
                let (align, size) = self.size_align(m, *e)?;
                let n = u32::try_from(*n).map_err(|_| "memory laid out in bytes of many elements")?;
                Ok((align, size.saturating_mul(n)))
            }
            Type::Struct { elements, packed, .. } => {
                let (mut align, mut at) = (1u32, 0u32);
                for &e in elements {
                    let (a, size) = self.size_align(m, e)?;
                    let a = if *packed { 1 } else { a };
                    align = align.max(a);
                    at = round_up(at, a).saturating_add(size);
                }
                Ok((align, round_up(at, align)))
            }
            other => Err(format!("memory laid out in bytes of {other:?}")),
        }
    }

    /// The byte offset of field `field` of a structure of `elements` (`packed` without padding).
    pub(crate) fn field_offset(
        &self,
        m: &Module,
        elements: &[TypeId],
        packed: bool,
        field: usize,
    ) -> Result<u32, String> {
        let mut at = 0u32;
        for (k, &e) in elements.iter().enumerate() {
            let (a, size) = self.size_align(m, e)?;
            at = if packed { at } else { round_up(at, a) };
            if k == field {
                return Ok(at);
            }
            at = at.saturating_add(size);
        }
        Err("a field past the structure".into())
    }

    /// The scalars of a value of LLVM type `ty` placed at byte `at`, in ascending byte order, appended to `out`.
    pub(crate) fn leaves(&self, m: &Module, ty: TypeId, at: u32, out: &mut Vec<Leaf>) -> Result<(), String> {
        let leaf = |kind: LeafKind| Leaf { offset: at, kind };
        if out.len() > MAX_LEAVES {
            return Err(format!(
                "memory laid out in bytes of more than {MAX_LEAVES} scalars"
            ));
        }
        match m.types.get(ty).ok_or_else(|| format!("type {ty}"))? {
            Type::Integer(1) => out.push(leaf(LeafKind::Bool)),
            Type::Integer(16) => out.push(leaf(LeafKind::Int16)),
            Type::Integer(32) => out.push(leaf(LeafKind::Int32)),
            Type::Integer(64) => out.push(leaf(LeafKind::Int64)),
            Type::Half => out.push(leaf(LeafKind::Float16)),
            Type::Float => out.push(leaf(LeafKind::Float32)),
            Type::Double => out.push(leaf(LeafKind::Float64)),
            Type::Integer(bits) => return Err(format!("memory laid out in bytes with an i{bits}")),
            Type::Array(n, e) | Type::Vector(n, e) => {
                let (_, size) = self.size_align(m, *e)?;
                let n = u32::try_from(*n).map_err(|_| "memory laid out in bytes of many elements")?;
                for k in 0..n {
                    self.leaves(m, *e, at.saturating_add(k.saturating_mul(size)), out)?;
                }
            }
            Type::Struct { elements, packed, .. } => {
                for (k, &e) in elements.iter().enumerate() {
                    let offset = self.field_offset(m, elements, *packed, k)?;
                    self.leaves(m, e, at.saturating_add(offset), out)?;
                }
            }
            other => return Err(format!("memory laid out in bytes of {other:?}")),
        }
        Ok(())
    }
}
