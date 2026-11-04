// fxc 10.0.26100.0 / dxc from Windows SDK 10.0.26100.0. CPU compilation only.
// fxc -T ps_5_1 -E main -D MODEL=51 -enable_unbounded_descriptor_tables unbounded.hlsl -Fo unbounded.51.dxbc
// dxc -T ps_6_0 -E main -D MODEL=60 unbounded.hlsl -Fo unbounded.60.dxil
#if MODEL == 50
#define BIND(r,s) register(r)
#else
#define BIND(r,s) register(r,s)
#endif
Texture2D<uint4> tex[] : register(t4, space2);
uint4 main(uint4 input : TEXCOORD0) : SV_Target { return tex[input.x].Load(int3(input.yz,0)); }
