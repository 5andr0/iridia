// ps_2_0
sampler2D Albedo : register(s0);
sampler2D Second : register(s1);
float4 Mix : register(c0);

float4 main(float4 colour : COLOR0, float2 uv : TEXCOORD0, float2 uv2 : TEXCOORD1) : COLOR0
{
    float4 a = tex2D(Albedo, uv);
    float4 b = tex2D(Second, uv2);
    return lerp(a, b, Mix.x) * colour + Mix.y;
}
