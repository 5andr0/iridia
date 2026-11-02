// Original fixture: layers laid over one another by weights, as a layered material's pixel program draws them.
// SDK 10.0.26100.0 fxc /T ps_5_0 /E over /Fo blend_over.dxbc blend.hlsl (and /E kinds below)
cbuffer Material : register(b0) { float3 tint; float sharpness; float adjust; };
Texture2D base_map : register(t0);
Texture2D layer_map : register(t1);
Texture2D top_map : register(t2);
Texture2D mask_map : register(t3);
SamplerState sampler0 : register(s0);
struct Input { float4 position : SV_Position; float2 uv : TEXCOORD0; float4 colour : COLOR0; };

float4 over(Input i) : SV_Target0 {
    float3 below = base_map.Sample(sampler0, i.uv).rgb;
    // a vertex lane sharpened against a mask, then moved part of the way there
    float c = saturate(i.colour.x);
    float m = 1 - mask_map.Sample(sampler0, i.uv * 8).x;
    float w1 = lerp(c, saturate(1 - sharpness * (2 * (1 - c) - m)), adjust);
    below = lerp(below, layer_map.Sample(sampler0, i.uv).rgb * 1.8, w1);
    // a vertex lane held between two bounds
    float w2 = min(max(i.colour.y, 0.25), 0.75);
    below = lerp(below, top_map.Sample(sampler0, i.uv).rgb, w2);
    return float4(saturate(below) * tint, 1);
}

// SDK 10.0.26100.0 fxc /T ps_5_0 /E kinds /Fo blend_kinds.dxbc blend.hlsl
// the same two registers added as floats and as integers
float4 kinds(Input i) : SV_Target0 {
    float a = i.colour.x;
    float b = i.colour.y;
    return float4(a + b, asfloat(asuint(a) + asuint(b)), 0, 1);
}
