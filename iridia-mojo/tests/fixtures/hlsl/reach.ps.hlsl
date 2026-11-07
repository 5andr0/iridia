// ps_3_0: a cube and a volume sample, the second colour, a loop left by a break
samplerCUBE Sky : register(s0);
sampler3D Volume : register(s1);
float4 Params : register(c0);

float4 main(float4 uv : TEXCOORD0, float3 dir : TEXCOORD1, float4 secondary : COLOR1) : COLOR0
{
    float4 acc = texCUBE(Sky, dir) + tex3D(Volume, uv.xyz) * secondary;
    [loop] for (int i = 0; i < 8; i++) {
        acc = acc * Params.x + Params.y;
        [branch] if (acc.x > Params.z) {
            break;
        }
    }
    return acc;
}
