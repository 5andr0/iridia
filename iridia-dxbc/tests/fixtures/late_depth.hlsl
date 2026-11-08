// The early_depth shader without the early-depth attribute: its UAV writes run before depth testing.
// Compile with fxc /T ps_5_0 /E main /Fo late_depth.dxbc late_depth.hlsl
RWBuffer<uint> marks : register(u1);


float4 main(float4 p : SV_Position) : SV_Target
{
    marks[uint(p.x)] = 1;
    return float4(p.xy, 0, 1);
}
