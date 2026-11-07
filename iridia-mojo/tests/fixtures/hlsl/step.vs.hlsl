// vs_2_0: step (sge, slt) of vectors and of one lane, the second colour, fog
float4x4 World : register(c0);
float4 Light : register(c4);

struct Out { float4 pos : POSITION; float4 colour : COLOR0; float4 secondary : COLOR1; float fog : FOG; };

Out main(float4 pos : POSITION, float3 normal : NORMAL)
{
    Out o;
    o.pos = mul(pos, World);
    float d = dot(normal, Light.xyz);
    o.colour = step(Light, pos);
    o.secondary = float4(step(0, d), step(d, Light.w), d * 0.5, 1);
    o.fog = o.pos.z * Light.w;
    return o;
}
