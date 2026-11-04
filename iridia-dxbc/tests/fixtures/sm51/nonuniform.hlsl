// fxc 10.0.26100.0 / dxc from Windows SDK 10.0.26100.0. CPU compilation only.
// fxc -T cs_5_1 -E main -D MODEL=51 -enable_unbounded_descriptor_tables nonuniform.hlsl -Fo nonuniform.51.dxbc
// dxc -T cs_6_0 -E main -D MODEL=60 nonuniform.hlsl -Fo nonuniform.60.dxil
#if MODEL == 50
#define BIND(r,s) register(r)
#else
#define BIND(r,s) register(r,s)
#endif
Texture2D<uint4> tex[3] : register(t4, space2);
RWBuffer<uint4> output : register(u3, space3);
[numthreads(4,1,1)] void main(uint3 id : SV_DispatchThreadID) { output[id.x] = tex[NonUniformResourceIndex(id.x%3)].Load(int3(0,0,0)); }
