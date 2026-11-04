// fxc 10.0.26100.0 / dxc from Windows SDK 10.0.26100.0. CPU compilation only.
// fxc -T ps_5_1 -E main -D MODEL=51 -enable_unbounded_descriptor_tables constants.hlsl -Fo constants.51.dxbc
// dxc -T ps_6_0 -E main -D MODEL=60 constants.hlsl -Fo constants.60.dxil
#if MODEL == 50
#define BIND(r,s) register(r)
#else
#define BIND(r,s) register(r,s)
#endif
struct Params { uint4 a; uint4 b[2]; };
ConstantBuffer<Params> params[3] : register(b4, space2);
uint4 main(uint4 input : TEXCOORD0) : SV_Target { return params[input.x % 3].a + params[(input.x+1)%3].b[input.y%2]; }
