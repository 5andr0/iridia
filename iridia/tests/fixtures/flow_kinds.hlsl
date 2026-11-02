// Fixture for the value graph's operand kinds, compiled to both bytecodes:
//   fxc /T ps_5_0 /E main /Fo flow_kinds.dxbc flow_kinds.hlsl
//   dxc -T ps_6_0 -E main -Fo flow_kinds.dxil flow_kinds.hlsl
// The same two values compared as floats, as unsigned integers and as signed integers: the same operands (a bitcast is
// no operation in the graph), three comparisons.
float4 main(float4 a : TEXCOORD0, float4 b : TEXCOORD1) : SV_Target0
{
    return float4(a.x < b.x ? 1.0 : 0.0, asuint(a.x) < asuint(b.x) ? 1.0 : 0.0, asint(a.x) < asint(b.x) ? 1.0 : 0.0, 0.0);
}
