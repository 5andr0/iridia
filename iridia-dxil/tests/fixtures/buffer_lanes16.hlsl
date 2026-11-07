// dxc -T cs_6_2 -enable-16bit-types -E main -Fo buffer_lanes16.dxil buffer_lanes16.hlsl
// 16-bit lanes of raw and structured buffers, at offsets that share and straddle 32-bit words (and 64-bit values,
// which dxc reads and writes as pairs of words). Its memory after one group on Direct3D 12 WARP is
// buffer_lanes16.warp.json, written by buffer_lanes16.warp.cpp.
struct S {
    half a;
    uint16_t b;
    half c;
    uint16_t d;
};
RWByteAddressBuffer raw : register(u0);
RWStructuredBuffer<S> st : register(u1);
RWByteAddressBuffer res : register(u2);

[numthreads(8, 1, 1)]
void main(uint i : SV_GroupIndex) {
    // reads: two 16-bit lanes from byte 2i (odd i straddles a word), three halves, a structure, a 64-bit lane
    uint16_t2 p = raw.Load<uint16_t2>(2 * i);
    half3 h = raw.Load<half3>(64 + 2 * i);
    S s = st[i];
    uint64_t w = raw.Load<uint64_t>(128 + 8 * i);
    double d = raw.Load<double>(160 + 8 * (i & 3));
    res.Store4(16 * i, uint4(uint(p.x) | (uint(p.y) << 16), asuint(float(h.x)), asuint(float(h.y) + float(h.z)),
                             uint(s.b) | (uint(s.d) << 16)));
    res.Store4(128 + 16 * i, uint4(uint(w), uint(w >> 32), asuint(float(s.a)), asuint(float(s.c) + float(d))));
    // writes: one 16-bit lane a thread (neighbours share a word), two halves at a straddling offset, a structure's
    // fields, a 64-bit lane
    raw.Store<uint16_t>(192 + 2 * i, uint16_t(0xa000 + i));
    raw.Store<half2>(226 + 4 * i, half2(half(i) * 0.5, -half(i)));
    st[i].b = uint16_t(3 * i + 1);
    st[i].c = half(i) + half(0.25);
    raw.Store<double>(272 + 8 * i, double(i) * 0.5);
}
