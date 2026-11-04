// fxc 10.0.26100.0 / dxc from Windows SDK 10.0.26100.0. CPU compilation only.
// fxc -T ps_5_1 -E main -D MODEL=51 -enable_unbounded_descriptor_tables samplers.hlsl -Fo samplers.51.dxbc
// dxc -T ps_6_0 -E main -D MODEL=60 samplers.hlsl -Fo samplers.60.dxil
#if MODEL == 50
#define BIND(r,s) register(r)
#else
#define BIND(r,s) register(r,s)
#endif
Texture2D<float4> tex : register(t4, space2);
SamplerState samplers[3] : register(s3, space3);
float4 main(uint4 input : TEXCOORD0) : SV_Target { return tex.SampleLevel(samplers[input.x%3],float2(1.25,0.75),0); }
