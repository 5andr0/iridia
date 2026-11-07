// A Shader Model 4.0 pixel program whose reflection holds minimum-precision constants and two constant buffers of
// one name. fxc stores min16float, min16int and min16uint as their own variable types (54, 56, 57) in 32 bits each,
// and keeps the two `cbuffer C` of namespace A as two buffers named A::C: it lists the bindings by register and the
// buffers in the order declared, so with -D SWAPPED (the first C at b9) the two orders differ. With -D ALIKE the two
// buffers are as large and their declarations alike, which leaves the bindings indistinguishable.
// Compiled with Microsoft fxc 10.0.26100.0:
//   fxc -T ps_4_0 -E main reflection_sm4.hlsl -Fo reflection_sm4.dxbc
//   fxc -T ps_4_0 -E main -D SWAPPED reflection_sm4.hlsl -Fo reflection_sm4.swapped.dxbc
//   fxc -T ps_4_0 -E main -D ALIKE reflection_sm4.hlsl -Fo reflection_sm4.alike.dxbc
min16float4 tint;
min16int2 offsets;
min16uint counts[4];

#ifdef SWAPPED
#define FIRST b9
#else
#define FIRST b1
#endif

namespace A {
    cbuffer C : register(FIRST) { float pad; float x; };
#ifdef ALIKE
    cbuffer C : register(b2) { float y; };
#else
    cbuffer C : register(b2) { float4 skip[2]; float y; };
#endif
}

float4 main(float4 at : SV_Position) : SV_Target0 {
    uint i = uint(at.x) & 3;
    return tint * float(offsets.x + offsets.y) + float(counts[i]) + A::x + A::y;
}
