#![allow(clippy::unwrap_used)]
//! Hand-assembled `vs_3_0` tokens for the address register: what the naga lift refuses of `mova`, relative float
//! constants and `sincos`, by name. Their answers are held on the IR, on both executors (iridia-d3d9-ir's
//! tests/vertex_address.rs).
use iridia_d3d9::{lift, read};

fn instruction(op: u32, operands: &[u32]) -> Vec<u32> {
    [vec![op | (operands.len() as u32) << 24], operands.to_vec()].concat()
}
fn program(body: &[u32]) -> Vec<u8> {
    [vec![0xfffe_0300], body.to_vec(), vec![0x0000_ffff]]
        .concat()
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect()
}
fn dcl(usage: u32, index: u32, dst: u32) -> Vec<u32> {
    instruction(31, &[0x8000_0000 | index << 16 | usage, dst])
}
/// `dcl_position v15` and `dcl_position o7`: the position is the program's one input and output.
fn position() -> Vec<u32> {
    [dcl(0, 0, 0x900f_000f), dcl(0, 0, 0xe00f_0007)].concat()
}
const A0_X: u32 = 0xb001_0000;
/// `v15.xxxx`.
const V15_X: u32 = 0x9000_000f;
/// `o7.xyzw`.
const O7: u32 = 0xe00f_0007;

fn refusal(body: &[u32]) -> String {
    lift(&program(body)).unwrap_err().to_string()
}

#[test]
fn what_the_address_register_cannot_do_is_refused_by_name() {
    let body = |tokens: &[u32]| [position(), instruction(46, &[A0_X, V15_X]), tokens.to_vec()].concat();
    // the relative address token must be a0 with one lane replicated
    for (address, why) in [
        (0xb0e4_0000u32, "does not replicate one lane"),
        // aL is register type 15: 7 in bits 28 to 30, 1 in bits 11 to 12
        (0xf000_0800, "aL is not supported"),
        (
            0xa000_0000,
            "where it is the address register a0 or the loop counter aL",
        ),
        (0xb100_0000, "has source modifier 1"),
    ] {
        let tokens = instruction(1, &[O7, 0xa0e4_2000, address]);
        assert!(
            refusal(&body(&tokens)).contains(why),
            "{address:#x}: {}",
            refusal(&body(&tokens))
        );
    }
    // the flag on a register that is not a float constant, and without its address token
    let tokens = instruction(1, &[O7, 0x80e4_2000, 0xb000_0000]);
    assert!(refusal(&body(&tokens)).contains("relative addressing (not supported)"));
    let tokens = instruction(1, &[O7, 0xa0e4_2000]);
    assert!(
        refusal(&body(&tokens)).contains("without its address token")
            || refusal(&body(&tokens)).contains("operands")
    );
    // a relative destination
    let tokens = instruction(1, &[O7 | 0x2000, 0x90e4_000f]);
    assert!(refusal(&body(&tokens)).contains("relative address token of a destination"));
    // a0 read before mova wrote it
    let no_mova = [position(), instruction(1, &[O7, 0xa0e4_2000, 0xb000_0000])].concat();
    assert!(refusal(&no_mova).contains("before any mova"));
    // mova's destination is a0 only, and only a vertex program has it
    let bad = [position(), instruction(46, &[0x800f_0000, V15_X])].concat();
    assert!(refusal(&bad).contains("Mova destination"));
    let pixel: Vec<u8> = [0xffff_0300u32]
        .into_iter()
        .chain(instruction(46, &[A0_X, 0x9000_0000]))
        .chain([0xffff])
        .flat_map(u32::to_le_bytes)
        .collect();
    assert!(
        read(&pixel)
            .unwrap_err()
            .to_string()
            .contains("is a vertex instruction, not one of ps_3_0")
    );
    // sincos writes x, y or xy only
    let bad = [position(), instruction(37, &[0x800f_0000, V15_X])].concat();
    assert!(refusal(&bad).contains("SINCOS writes only"));
}
