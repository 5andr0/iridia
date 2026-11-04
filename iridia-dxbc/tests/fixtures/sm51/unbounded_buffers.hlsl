// fxc 10.0.26100.0 / dxc from Windows SDK 10.0.26100.0. CPU compilation only.
// fxc -T cs_5_1 -E main -D MODEL=51 -enable_unbounded_descriptor_tables unbounded_buffers.hlsl -Fo unbounded_buffers.51.dxbc
// dxc -T cs_6_0 -E main -D MODEL=60 unbounded_buffers.hlsl -Fo unbounded_buffers.60.dxil
#if MODEL == 50
#define BIND(r,s) register(r)
#else
#define BIND(r,s) register(r,s)
#endif
cbuffer Params : register(b3, space1) { uint which; };
ByteAddressBuffer input[] : register(t4, space2);
RWByteAddressBuffer output[] : register(u4, space3);
[numthreads(1,1,1)] void main(uint3 id : SV_DispatchThreadID) { output[which+1].Store4(id.x*16, input[which].Load4(id.x*16) + uint4(1,2,3,4)); }
