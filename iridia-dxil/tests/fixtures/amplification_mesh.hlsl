// The mesh program amplification.hlsl launches (Shader Model 6.5): each thread group writes the quad's four corners in
// the payload's colour and one of its two triangles, the one its group id picks. Compiled with
//   dxc -T ms_6_5 -E main (amplification_mesh.dxil).
struct Payload {
    float4 color;
};

struct Vertex {
    float4 position : SV_Position;
    float4 color : COLOR;
};

static const float4 positions[4] = {
    float4(-1, -1, 0, 1), float4(-1, 1, 0, 1), float4(1, 1, 0, 1), float4(1, -1, 0, 1),
};
static const uint3 triangles[2] = { uint3(0, 1, 3), uint3(1, 2, 3) };

[outputtopology("triangle")]
[numthreads(4, 1, 1)]
void main(in payload Payload p, out vertices Vertex vertices[4], out indices uint3 indices[1],
          uint index : SV_GroupIndex, uint3 group : SV_GroupID) {
    SetMeshOutputCounts(4, 1);
    vertices[index].position = positions[index];
    vertices[index].color = p.color;
    if (index == 0) {
        indices[0] = triangles[group.x];
    }
}
