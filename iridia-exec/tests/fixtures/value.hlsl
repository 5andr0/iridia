// Fixture for iridia-exec's value evaluator, compiled to both bytecodes:
//   fxc /T ps_5_0 /E main /Fo value.dxbc value.hlsl
//   dxc -T ps_6_0 -E main -Fo value.dxil value.hlsl
// A palette factor's shape: a record number rounded from an input lane and converted to an integer, a slot converted
// from a constant, a row of a constant-buffer array read under a bound; a texel blended toward a colour and saturated;
// and the same two values compared as floats and as unsigned integers.
cbuffer Instances : register(b4)
{
    float4 palette[26];
};
cbuffer Material : register(b1)
{
    float4 slot;
    float4 tint;
};
Texture2D albedo : register(t0);
SamplerState wrap : register(s0);

struct Input
{
    float4 position : SV_Position;
    float4 normal : NORMAL0;
    float2 uv : TEXCOORD0;
};

float4 main(Input i) : SV_Target0
{
    uint record = (uint)round(i.normal.w);
    uint index = record * 13 + 9 + (uint)slot.w;
    float4 factor = index < 26 ? palette[index] : float4(0, 0, 0, 0);
    float4 t = albedo.Sample(wrap, i.uv);
    float3 colour = saturate(lerp(t.rgb, tint.rgb, tint.a)) * factor.rgb;
    float kinds = (i.normal.x < i.normal.y ? 1.0f : 0.0f) + (asuint(i.normal.x) < asuint(i.normal.y) ? 2.0f : 0.0f);
    return float4(colour, min(max(i.normal.z, tint.x), 2.0) + kinds);
}
