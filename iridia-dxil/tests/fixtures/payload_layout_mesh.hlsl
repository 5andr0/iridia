// The mesh program payload_layout.hlsl launches (Shader Model 6.6, native 16-bit types): it reads the payload as the
// 16 halfwords it is, and writes them as the outputs of one triangle's three vertices. Compiled with
//   dxc -T ms_6_6 -E main -enable-16bit-types (payload_layout_mesh.dxil).
struct Payload {
    uint16_t words[16];
};

struct Vertex {
    float4 position : SV_Position;
    nointerpolation uint4 a : A;
    nointerpolation uint4 b : B;
    nointerpolation uint4 c : C;
    nointerpolation uint4 d : D;
};

[outputtopology("triangle")]
[numthreads(3, 1, 1)]
void main(in payload Payload p, out vertices Vertex vertices[3], out indices uint3 indices[1],
          uint index : SV_GroupIndex) {
    SetMeshOutputCounts(3, 1);
    vertices[index].position = float4(index, 0, 0, 1);
    vertices[index].a = uint4(p.words[0], p.words[1], p.words[2], p.words[3]);
    vertices[index].b = uint4(p.words[4], p.words[5], p.words[6], p.words[7]);
    vertices[index].c = uint4(p.words[8], p.words[9], p.words[10], p.words[11]);
    vertices[index].d = uint4(p.words[12], p.words[13], p.words[14], p.words[15]);
    if (index == 0) {
        indices[0] = uint3(0, 1, 2);
    }
}
