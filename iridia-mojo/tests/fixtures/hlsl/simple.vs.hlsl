// vs_2_0
float4x4 WorldViewProjection : register(c0);
float4 Offset : register(c4);

struct Out { float4 position : POSITION; float4 colour : COLOR0; float2 uv : TEXCOORD0; float2 uv2 : TEXCOORD1; };

Out main(float4 position : POSITION, float4 colour : COLOR0, float2 uv : TEXCOORD0)
{
    Out o;
    o.position = mul(position, WorldViewProjection);
    o.colour = colour * Offset.x;
    o.uv = uv;
    o.uv2 = uv * Offset.yz + Offset.w;
    return o;
}
