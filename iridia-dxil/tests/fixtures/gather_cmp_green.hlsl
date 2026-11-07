// A comparing gather of a depth texture's green channel, after vkd3d's gather-cmp.shader_test (GatherCmpGreen of an
// r32 shadow view). WGSL gathers a depth texture's one lane, so the lifters refuse it by name. Compiled with
//   dxc -T ps_6_0 -E main (iridia-dxil/tests/fixtures/gather_cmp_green.dxil)
//   fxc /T ps_5_0 /E main (iridia-dxbc/tests/fixtures/gather_cmp_green.dxbc)
Texture2D<float> t : register(t0);
SamplerComparisonState s : register(s0);

float4 main(float4 position : SV_Position) : SV_Target
{
    return t.GatherCmpGreen(s, position.xy / 640.0, 0.5);
}
