// fxc 10.0.26100.0 / dxc from Windows SDK 10.0.26100.0. CPU compilation only.
// fxc -T cs_5_1 -E main -D MODEL=51 -enable_unbounded_descriptor_tables images.hlsl -Fo images.51.dxbc
// dxc -T cs_6_0 -E main -D MODEL=60 images.hlsl -Fo images.60.dxil
#if MODEL == 50
#define BIND(r,s) register(r)
#else
#define BIND(r,s) register(r,s)
#endif
cbuffer Params : register(b3, space1) { uint which; };
RWTexture2D<uint4> output[3] : register(u4, space2);
[numthreads(1,1,1)] void main(uint3 id : SV_DispatchThreadID) { output[which%3][id.xy] = uint4(id, which+17); }
