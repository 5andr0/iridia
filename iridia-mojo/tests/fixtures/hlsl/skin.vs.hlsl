// vs_3_0: matrices, relative addressing through a bone index, varyings
float4x4 ViewProjection : register(c0);
float4x3 Bones[20] : register(c8);
float4 Light : register(c4);

struct Out {
    float4 position : POSITION;
    float4 colour : COLOR0;
    float4 uv : TEXCOORD0;
    float3 normal : TEXCOORD3;
    float fog : FOG;
};

Out main(float4 position : POSITION, float3 normal : NORMAL, float4 uv : TEXCOORD0, float4 index : BLENDINDICES,
         float4 colour : COLOR0)
{
    Out o;
    float3 world = mul(position, Bones[index.x]);
    float3 n = mul(normal, (float3x3)Bones[index.x]);
    o.position = mul(float4(world, 1), ViewProjection);
    float l = saturate(dot(normalize(n), Light.xyz));
    o.colour = colour * l;
    float s, c;
    sincos(uv.x, s, c);
    o.uv = float4(uv.xy, s, c);
    o.normal = cross(n, Light.xyz);
    o.fog = length(world) * Light.w;
    return o;
}
