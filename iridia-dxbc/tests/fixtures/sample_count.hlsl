// The render target's sample count is the pipeline's (its rasterizer state), not the program's: iridia's IR lift
// refuses it by name. Compiled with fxc /T ps_5_0 /E main.
float4 main() : SV_Target
{
    return float4((float)GetRenderTargetSampleCount(), 0, 0, 1);
}
