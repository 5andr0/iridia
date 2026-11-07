// What a ray query reports of the ray in an instance's object space, of a procedural candidate's opacity and of the ray
// it was traced with (compute, one group of 64). Compiled with dxc -T cs_6_5 -E main (ray_query_object.dxil). Thread i
// traces the ray origins[i], directions[i] (t from 0 to 1000) twice, and writes 132 words from results[132 * i], every
// float as its bits.
//
// The first query, declared with RAY_FLAG_SKIP_CLOSEST_HIT_SHADER and traced with the flags and instance mask of
// params[i].xy, commits as ray_query_procedural.hlsl does, but intersects the sphere with the candidate's object-space
// ray. Words 0 to 8: the committed status, the query's ray flags, the world ray's origin and direction, its t min; then
// for a committed hit, words 9 to 42: the ray's origin and direction in the object space of the hit's instance, the
// instance's object-to-world and world-to-object matrices (three rows of four, row by row), the hit's distance, and its
// instance, geometry and primitive indices.
//
// The second, declared with RAY_FLAG_SKIP_TRIANGLES and traced with no flags and the same mask, commits nothing, so every
// procedural primitive the ray reaches is offered, in whatever order: for primitive p of instance k, the seven words from
// 48 + 7 * (4 * k + p) are 1 plus whether it is non-opaque, then the ray's origin and direction in the instance's object
// space.
RaytracingAccelerationStructure scene : register(t0);
RWStructuredBuffer<uint> results : register(u0);

cbuffer Rays : register(b0) {
    float4 origins[64];
    float4 directions[64];
    uint4 params[64];
    float4 spheres[8];
};

void put3(uint at, float3 v) {
    results[at] = asuint(v.x);
    results[at + 1] = asuint(v.y);
    results[at + 2] = asuint(v.z);
}

void put3x4(uint at, float3x4 m) {
    for (uint r = 0; r < 3; r++) {
        for (uint c = 0; c < 4; c++) {
            results[at + 4 * r + c] = asuint(m[r][c]);
        }
    }
}

[numthreads(64, 1, 1)]
void main(uint i : SV_GroupIndex) {
    RayDesc ray;
    ray.Origin = origins[i].xyz;
    ray.Direction = directions[i].xyz;
    ray.TMin = 0.0;
    ray.TMax = 1000.0;
    uint at = 132 * i;

    RayQuery<RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> q;
    q.TraceRayInline(scene, params[i].x, params[i].y, ray);
    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
            q.CommitNonOpaqueTriangleHit();
        } else {
            float3 o = q.CandidateObjectRayOrigin();
            float3 d = q.CandidateObjectRayDirection();
            float4 sphere = spheres[q.CandidatePrimitiveIndex() & 7];
            float3 m = o - sphere.xyz;
            float a = dot(d, d);
            float b = dot(m, d);
            float c = dot(m, m) - sphere.w * sphere.w;
            float disc = b * b - a * c;
            if (disc >= 0.0) {
                float t = (-b - sqrt(disc)) / a;
                if (t >= q.RayTMin() && (q.CommittedStatus() == COMMITTED_NOTHING || t < q.CommittedRayT())) {
                    q.CommitProceduralPrimitiveHit(t);
                }
            }
        }
    }
    uint status = q.CommittedStatus();
    results[at] = status;
    results[at + 1] = q.RayFlags();
    put3(at + 2, q.WorldRayOrigin());
    put3(at + 5, q.WorldRayDirection());
    results[at + 8] = asuint(q.RayTMin());
    if (status != COMMITTED_NOTHING) {
        put3(at + 9, q.CommittedObjectRayOrigin());
        put3(at + 12, q.CommittedObjectRayDirection());
        put3x4(at + 15, q.CommittedObjectToWorld3x4());
        put3x4(at + 27, q.CommittedWorldToObject3x4());
        results[at + 39] = asuint(q.CommittedRayT());
        results[at + 40] = q.CommittedInstanceIndex();
        results[at + 41] = q.CommittedGeometryIndex();
        results[at + 42] = q.CommittedPrimitiveIndex();
    }

    RayQuery<RAY_FLAG_SKIP_TRIANGLES> p;
    p.TraceRayInline(scene, 0, params[i].y, ray);
    while (p.Proceed()) {
        uint slot = at + 48 + 7 * (4 * min(p.CandidateInstanceIndex(), 2) + (p.CandidatePrimitiveIndex() & 3));
        results[slot] = p.CandidateProceduralPrimitiveNonOpaque() ? 2 : 1;
        put3(slot + 1, p.CandidateObjectRayOrigin());
        put3(slot + 4, p.CandidateObjectRayDirection());
    }
}
