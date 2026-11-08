// The vertex program of sampler_heap.warp.cpp: one clockwise triangle that covers the pixel, the pixel program's TEXCOORD0
// from row 1 of `Case`. Compiled with dxc -T vs_6_0 -E main (sampler_heap.vs.dxil).
cbuffer Case : register(b0, space1) {
    uint4 r[32];
};

struct Out {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

Out main(uint id : SV_VertexID) {
    Out o;
    float2 p = float2(id == 1 ? 3.0 : -1.0, id == 2 ? -3.0 : 1.0);
    o.position = float4(p, 0.5, 1.0);
    o.uv = asfloat(r[1].xy);
    return o;
}
