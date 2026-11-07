// A quad patch whose patch-constant function leaves two of its patch constants unwritten (UNWRITTEN, its TANGENT
// array): fxc declares no output register for them in any fork or join phase, yet the patch-constant signature places
// them (o7 to o10). The factors are constants, so fxc writes them from an immediate constant buffer through an indexed
// output (`mov o[r0.x + 0].x, icb[r0.x + 0].y`), as the dxc test hs.pcf.void.hlsl compiles. Compiled with
//   fxc /T vs_5_0 /E vs (hull_unwritten.vs.dxbc),  fxc /T hs_5_0 /E hs (hull_unwritten.hs.dxbc),
//   fxc /T ds_5_0 /E ds (hull_unwritten.ds.dxbc),  fxc /T gs_5_0 /E gs (hull_unwritten.gs.dxbc)
// (fxc warns X3578, an output value not completely initialized, for the patch constants left out).
struct CP {
    float4 p : POSITION;
};
struct PC {
    float e[4] : SV_TessFactor;
    float i[2] : SV_InsideTessFactor;
    float4 extra : EXTRA;
    float4 unwritten : UNWRITTEN;
    float3 tangent[3] : TANGENT;
};
struct DO {
    float4 loc : LOC;
    float4 mix : MIX;
};

CP vs(uint id : SV_VertexID) {
    CP o;
    o.p = float4(id + 1, 2 * id + 1, 3, 4);
    return o;
}

PC pc(InputPatch<CP, 4> patch, uint id : SV_PrimitiveID) {
    PC o;
    o.e[0] = 1.0;
    o.e[1] = 2.0;
    o.e[2] = 3.0;
    o.e[3] = 4.0;
    o.i[0] = 5.0;
    o.i[1] = 6.0;
    o.extra = patch[0].p + patch[3].p + float4(id, 0, 0, 0);
    return o;
}

[domain("quad")]
[partitioning("fractional_odd")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(4)]
[patchconstantfunc("pc")]
CP hs(InputPatch<CP, 4> patch, uint i : SV_OutputControlPointID) {
    CP o = patch[i];
    o.p += float4(i, 0.5 * i, 0.25, 0);
    return o;
}

[domain("quad")]
DO ds(PC pc, float2 loc : SV_DomainLocation, const OutputPatch<CP, 4> patch) {
    DO o;
    o.loc = float4(loc, 0, 1);
    o.mix = float4(loc.x * patch[1].p.x + pc.extra.x, loc.y * patch[2].p.y + pc.extra.y, pc.extra.z, pc.extra.w);
    return o;
}

[maxvertexcount(3)]
void gs(triangle DO v[3], inout TriangleStream<DO> s) {
    for (uint i = 0; i < 3; i++) s.Append(v[i]);
}
