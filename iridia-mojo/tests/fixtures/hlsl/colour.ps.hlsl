// ps_3_0: samples, the arithmetic MojoShader writes as GLSL functions, a discard and two targets
sampler2D Albedo : register(s0);
sampler2D Detail : register(s3);
float4 Tint : register(c0);
float4 Params : register(c1);
float4 Fog : register(c4);

struct Out { float4 colour : COLOR0; float4 extra : COLOR1; };

Out main(float4 uv : TEXCOORD0, float4 colour : COLOR0, float3 normal : TEXCOORD2)
{
    Out o;
    float4 a = tex2D(Albedo, uv.xy);
    float4 d = tex2D(Detail, uv.zw * Params.x);
    clip(a.w - Params.y);
    float3 n = normalize(normal);
    float light = saturate(dot(n, Fog.xyz));
    float3 c = lerp(a.rgb, d.rgb, Params.z) * colour.rgb * Tint.rgb;
    c = pow(abs(c), Params.w) + frac(uv.x * 7.0) * 0.01;
    float s = sign(normal.x) * rsqrt(abs(normal.y) + 1.0) + 1.0 / (abs(normal.z) + 0.5);
    o.colour = float4(c * light, a.w * Tint.w);
    o.extra = float4(min(c.r, s), max(c.g, light), exp2(uv.y), log2(abs(uv.x) + 1.0));
    o.extra.x += (uv.x > uv.y) ? 1.0 : -1.0;
    return o;
}
