// The render target's sample count (GetRenderTargetSampleCount) is the pipeline's, its rasterizer state, not the
// program's: every lifter refuses it by name, and goes on refusing it (a program that reads it is a stable refusal for
// a caller's tests). Compiled with
//   dxc -T ps_6_0 -E main (iridia-dxil/tests/fixtures/sample_count.dxil)
//   fxc /T ps_5_0 /E main (iridia-dxbc/tests/fixtures/sample_count.dxbc)
float4 main() : SV_Target
{
    return float4((float)GetRenderTargetSampleCount(), 0, 0, 1);
}
