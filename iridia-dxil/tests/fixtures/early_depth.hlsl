// A pixel program that forces Direct3D's depth and stencil tests before it runs (`[earlydepthstencil]`), so its UAV
// store is made only where they pass. Compile:
//   dxc -T ps_6_0 -E main -Fo early_depth.dxil early_depth.hlsl   (iridia-dxil)
//   fxc /T ps_5_0 /E main /Fo early_depth.dxbc early_depth.hlsl   (iridia-dxbc)
RWBuffer<uint> marks : register(u1);

[earlydepthstencil]
float4 main(float4 p : SV_Position) : SV_Target
{
    marks[uint(p.x)] = 1;
    return float4(p.xy, 0, 1);
}
