// An amplification program (Shader Model 6.6, native 16-bit types) whose groupshared payload has two halfword fields in
// one word, which two threads write at once: neither write is lost, the word is 0xbbbbaaaa. Compiled with
//   dxc -T as_6_6 -E main -enable-16bit-types (payload_halves.dxil).
struct Payload {
    uint16_t low;
    uint16_t high;
    uint rest;
};

groupshared Payload payload;

[numthreads(2, 1, 1)]
void main(uint index : SV_GroupIndex) {
    if (index == 0) {
        payload.low = 0xaaaa;
    } else {
        payload.high = 0xbbbb;
    }
    payload.rest = 7;
    DispatchMesh(1, 1, 1, payload);
}
