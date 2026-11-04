// fxc 10.0.26100.0 / dxc from Windows SDK 10.0.26100.0. CPU compilation only.
// fxc -T cs_5_1 -E main -D MODEL=51 -enable_unbounded_descriptor_tables spaces.hlsl -Fo spaces.51.dxbc
// dxc -T cs_6_0 -E main -D MODEL=60 spaces.hlsl -Fo spaces.60.dxil
// fxc -T cs_5_0 -E main -D MODEL=50 spaces.hlsl -Fo spaces.50.dxbc
#if MODEL == 50
#define BIND(r,s) register(r)
#else
#define BIND(r,s) register(r,s)
#endif
cbuffer Params : BIND(b4, space1) { uint4 bias; };
Texture2D<float4> tex : BIND(t3, space2);
SamplerState samp : BIND(s2, space3);
Texture2D<float> depth_tex : BIND(t6, space2);
SamplerComparisonState cmp : BIND(s5, space3);
Buffer<uint4> typed : BIND(t8, space2);
ByteAddressBuffer raw : BIND(t9, space2);
struct Item { uint4 v; };
StructuredBuffer<Item> structured : BIND(t10, space2);
RWTexture2D<uint4> image_out : BIND(u2, space4);
RWBuffer<uint4> typed_out : BIND(u3, space4);
RWByteAddressBuffer raw_out : BIND(u4, space4);
RWStructuredBuffer<Item> structured_out : BIND(u5, space4);
[numthreads(1,1,1)] void main(uint3 id : SV_DispatchThreadID) {
 uint4 v = bias + typed.Load(id.x) + raw.Load4(id.x * 16) + structured[id.x].v;
 v += asuint(tex.SampleLevel(samp, float2(0.25,0.75),0));
 v.x += asuint(depth_tex.SampleCmpLevelZero(cmp, float2(0.25,0.75),0.5));
 image_out[id.xy] = v; typed_out[id.x] = v; raw_out.Store4(id.x*16,v); structured_out[id.x].v = v;
}
