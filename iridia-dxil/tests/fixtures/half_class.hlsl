// The class of a half and the conversion of a float to a 16-bit integer (pixel). Compiled with dxc -T ps_6_2
// -enable-16bit-types -E main (half_class.dxil, also -Od), and with -T ps_6_9 (half_class.69.dxil, also -Od), where dxc writes isnormal
// of a half as dx.op.isSpecialFloat.f16 (IsNormal; the validator allows a 16-bit overload from shader model 6.9) and not as
// bit tests of the exponent.
//   SV_Target0 = isnormal of the halves of v: 1 where the half is normal (its exponent is neither 0 nor 31), 0 where it
//   is zero or subnormal. v = (2^-14, 2^-15, 1.0, 0.0) gives (1, 0, 1, 0): 2^-14 is the smallest normal half, 2^-15 a
//   subnormal one (a normal float).
//   SV_Target1 = the 16-bit integers (int16_t) w converts to, back as floats: w = (-3.7, 100.9, -32768.0, 32767.0) gives
//   (-3, 100, -32768, 32767).
cbuffer Values : register(b0) {
    float4 v;
    float4 w;
};

struct Outputs {
    float4 normal : SV_Target0;
    float4 integer : SV_Target1;
};

Outputs main(float4 position : SV_Position) {
    Outputs o;
    half4 h = (half4)v;
    o.normal = float4(isnormal(h.x), isnormal(h.y), isnormal(h.z), isnormal(h.w));
    o.integer = float4((int16_t)w.x, (int16_t)w.y, (int16_t)w.z, (int16_t)w.w);
    return o;
}
