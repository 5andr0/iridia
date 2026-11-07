// An amplification program (Shader Model 6.6, native 16-bit types) whose payload is a local structure of fields of
// every size: Direct3D lays it out as C does, each field at its natural alignment and a bool four bytes, so `before`
// is at byte 0, `wide` at 8, `after` at 16, `flag` at 20 and `small` at 24, 32 bytes in all. Compiled with
//   dxc -T as_6_6 -E main -enable-16bit-types (payload_layout.dxil).
struct Payload {
    uint16_t before;
    uint64_t wide;
    uint16_t after;
    bool flag;
    half small;
};

[numthreads(1, 1, 1)]
void main() {
    Payload p;
    p.before = 0xabab;
    p.wide = 0x1122334455667788;
    p.after = 0xcdcd;
    p.flag = true;
    p.small = 1.5;
    DispatchMesh(1, 1, 1, p);
}
