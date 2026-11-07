// ps_3_0: a constant-count loop (rep), a branch on a computed value, derivatives
float4 Steps : register(c0);
float4 Scale : register(c2);

float4 main(float4 uv : TEXCOORD0, float4 colour : COLOR0) : COLOR0
{
    float4 acc = 0;
    [loop] for (int i = 0; i < 5; i++) {
        acc += Steps * (uv.x + i);
        acc.yzw = acc.yzw * 0.5;
    }
    [branch] if (acc.x > Scale.x) {
        acc.xy = acc.yx;
    } else {
        acc = acc * Scale + colour;
    }
    float2 dx = ddx(uv.xy);
    float2 dy = ddy(uv.xy);
    acc.zw += dx * Scale.z + dy * Scale.w;
    return acc;
}
