// fxc 10.0.26100.0 / dxc from Windows SDK 10.0.26100.0. CPU compilation only.
// fxc -T ps_5_1 -E main -D MODEL=51 -enable_unbounded_descriptor_tables unbounded_constants.hlsl -Fo unbounded_constants.51.dxbc
// dxc -T ps_6_0 -E main -D MODEL=60 unbounded_constants.hlsl -Fo unbounded_constants.60.dxil
#if MODEL == 50
#define BIND(r,s) register(r)
#else
#define BIND(r,s) register(r,s)
#endif
struct Params { uint4 v; };
ConstantBuffer<Params> params[] : register(b4, space2);
uint4 main(uint4 input : TEXCOORD0) : SV_Target { return params[input.x].v; }
