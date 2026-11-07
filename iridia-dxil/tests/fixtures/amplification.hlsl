// An amplification program (Shader Model 6.5) after vkd3d's mesh.shader_test: four threads each write one lane of a
// groupshared payload, and DispatchMesh launches 2 mesh thread groups from thread 0, 1 from the others. Compiled with
//   dxc -T as_6_5 -E main (amplification.dxil).
// The launch that counts is the first thread's (two groups), and DispatchMesh's barrier collects every thread's write:
// the payload is (0.25, 0.5, 0.75, 1).
struct Payload {
    float4 color;
};

groupshared Payload payload;

[numthreads(4, 1, 1)]
void main(uint index : SV_GroupIndex) {
    switch (index) {
        case 0: payload.color.x = 0.25; break;
        case 1: payload.color.y = 0.5; break;
        case 2: payload.color.z = 0.75; break;
        case 3: payload.color.w = 1.0; break;
    }
    uint x = 1;
    if (index == 0) {
        x = 2;
    }
    DispatchMesh(x, 1, 1, payload);
}
