//! Inflate (RFC 1951) and gzip (RFC 1952): the committed case files are gzipped JSON, and PNG data is a zlib stream.

struct Bits<'a> {
    s: &'a [u8],
    pos: usize,
    bit: u32,
}

impl Bits<'_> {
    fn bit(&mut self) -> Result<u32, String> {
        let byte = *self.s.get(self.pos).ok_or("inflate: input ends")?;
        let b = u32::from(byte >> self.bit) & 1;
        self.bit += 1;
        if self.bit == 8 {
            self.bit = 0;
            self.pos += 1;
        }
        Ok(b)
    }

    fn bits(&mut self, n: u32) -> Result<u32, String> {
        let mut v = 0;
        for k in 0..n {
            v |= self.bit()? << k;
        }
        Ok(v)
    }
}

/// A canonical Huffman code: counts per length and the symbols in code order.
struct Huffman {
    counts: [u16; 16],
    symbols: Vec<u16>,
}

impl Huffman {
    fn new(lengths: &[u8]) -> Huffman {
        let mut counts = [0u16; 16];
        for &l in lengths {
            counts[usize::from(l)] += 1;
        }
        counts[0] = 0;
        let mut offs = [0u16; 16];
        for l in 1..16 {
            offs[l] = offs[l - 1] + counts[l - 1];
        }
        let mut symbols = vec![0u16; lengths.len()];
        for (s, &l) in lengths.iter().enumerate() {
            if l != 0 {
                symbols[usize::from(offs[usize::from(l)])] = s as u16;
                offs[usize::from(l)] += 1;
            }
        }
        Huffman { counts, symbols }
    }

    fn decode(&self, b: &mut Bits) -> Result<u16, String> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..16 {
            code |= b.bit()? as i32;
            let count = i32::from(self.counts[len]);
            if code - count < first {
                return Ok(self.symbols[(index + (code - first)) as usize]);
            }
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        Err("inflate: bad code".into())
    }
}

const LEN_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195,
    227, 258,
];
const LEN_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073,
    4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
];

/// Inflate a raw deflate stream.
pub fn inflate(s: &[u8], size_hint: usize) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(size_hint);
    let mut b = Bits { s, pos: 0, bit: 0 };
    loop {
        let last = b.bit()?;
        match b.bits(2)? {
            0 => {
                if b.bit != 0 {
                    b.bit = 0;
                    b.pos += 1;
                }
                let at = b.pos;
                let len = usize::from(u16::from_le_bytes([
                    *s.get(at).ok_or("inflate: ends")?,
                    *s.get(at + 1).ok_or("inflate: ends")?,
                ]));
                let block = s.get(at + 4..at + 4 + len).ok_or("inflate: stored block ends")?;
                out.extend_from_slice(block);
                b.pos = at + 4 + len;
            }
            kind @ (1 | 2) => {
                let (lit, dist) = if kind == 1 {
                    let mut l = [8u8; 288];
                    l[144..256].fill(9);
                    l[256..280].fill(7);
                    (Huffman::new(&l), Huffman::new(&[5u8; 30]))
                } else {
                    let hlit = b.bits(5)? as usize + 257;
                    let hdist = b.bits(5)? as usize + 1;
                    let hclen = b.bits(4)? as usize + 4;
                    const ORDER: [usize; 19] =
                        [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
                    let mut cl = [0u8; 19];
                    for &o in &ORDER[..hclen] {
                        cl[o] = b.bits(3)? as u8;
                    }
                    let clh = Huffman::new(&cl);
                    let mut lengths = Vec::with_capacity(hlit + hdist);
                    while lengths.len() < hlit + hdist {
                        let sym = clh.decode(&mut b)?;
                        match sym {
                            0..=15 => lengths.push(sym as u8),
                            16 => {
                                let prev = *lengths.last().ok_or("inflate: repeat with no length")?;
                                for _ in 0..3 + b.bits(2)? {
                                    lengths.push(prev);
                                }
                            }
                            17 => lengths.extend(std::iter::repeat_n(0, 3 + b.bits(3)? as usize)),
                            _ => lengths.extend(std::iter::repeat_n(0, 11 + b.bits(7)? as usize)),
                        }
                    }
                    (Huffman::new(&lengths[..hlit]), Huffman::new(&lengths[hlit..]))
                };
                loop {
                    let sym = lit.decode(&mut b)?;
                    match sym {
                        0..=255 => out.push(sym as u8),
                        256 => break,
                        _ => {
                            let i = usize::from(sym - 257);
                            let len = usize::from(*LEN_BASE.get(i).ok_or("inflate: length")?)
                                + b.bits(u32::from(LEN_EXTRA[i]))? as usize;
                            let d = usize::from(dist.decode(&mut b)?);
                            let distance = usize::from(*DIST_BASE.get(d).ok_or("inflate: distance")?)
                                + b.bits(u32::from(DIST_EXTRA[d]))? as usize;
                            if distance > out.len() {
                                return Err("inflate: distance before the start".into());
                            }
                            let from = out.len() - distance;
                            for k in 0..len {
                                out.push(out[from + k]);
                            }
                        }
                    }
                }
            }
            _ => return Err("inflate: block type 3".into()),
        }
        if last == 1 {
            return Ok(out);
        }
    }
}

/// The contents of a gzip file (one member), its CRC and length checked.
pub fn gunzip(file: &[u8]) -> Result<Vec<u8>, String> {
    if file.len() < 18 || file[0] != 0x1f || file[1] != 0x8b || file[2] != 8 {
        return Err("not a gzip file".into());
    }
    let flags = file[3];
    let mut at = 10;
    if flags & 4 != 0 {
        let n = usize::from(u16::from_le_bytes([file[at], file[at + 1]]));
        at += 2 + n;
    }
    for bit in [8u8, 16] {
        if flags & bit != 0 {
            at += file
                .get(at..)
                .and_then(|r| r.iter().position(|&c| c == 0))
                .ok_or("gzip: header ends")?
                + 1;
        }
    }
    if flags & 2 != 0 {
        at += 2;
    }
    let tail = file.len() - 8;
    let size = u32::from_le_bytes(file[tail + 4..].try_into().map_err(|_| "gzip: trailer")?) as usize;
    let out = inflate(file.get(at..tail).ok_or("gzip: ends")?, size)?;
    let crc = u32::from_le_bytes(file[tail..tail + 4].try_into().map_err(|_| "gzip: trailer")?);
    if crc32(&out) != crc || out.len() as u32 != size as u32 {
        return Err("gzip: the CRC or the length does not match".into());
    }
    Ok(out)
}

/// CRC-32 (IEEE 802.3), as gzip checks it.
fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, t) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 {
                0xedb8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
        }
        *t = c;
    }
    let mut c = !0u32;
    for &b in data {
        c = table[((c ^ u32::from(b)) & 0xff) as usize] ^ (c >> 8);
    }
    !c
}

/// A file's text, gunzipped when its name ends in `.gz`.
pub fn read_text(path: &std::path::Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let bytes = if path.extension().is_some_and(|e| e == "gz") {
        gunzip(&bytes).map_err(|e| format!("{}: {e}", path.display()))?
    } else {
        bytes
    };
    String::from_utf8(bytes).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inflates_stored_and_fixed_blocks() {
        // "hello" stored, then the same deflated with fixed codes (zlib.compress(b"hello")[2:-4])
        let stored = [1u8, 5, 0, 0xfa, 0xff, b'h', b'e', b'l', b'l', b'o'];
        assert_eq!(inflate(&stored, 5).expect("inflates"), b"hello");
        let fixed = [0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x07, 0x00];
        assert_eq!(inflate(&fixed, 5).expect("inflates"), b"hello");
    }

    #[test]
    fn gunzips() {
        // gzip.compress(b"hello", mtime=0)
        let gz = [
            0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 2, 0xff, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x07, 0x00, 0x86, 0xa6,
            0x10, 0x36, 5, 0, 0, 0,
        ];
        assert_eq!(gunzip(&gz).expect("gunzips"), b"hello");
    }
}
