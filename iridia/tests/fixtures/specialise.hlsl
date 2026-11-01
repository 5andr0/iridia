// Fixtures for tests/specialise.rs, each entry point compiled to both bytecodes:
//   fxc /T ps_5_0 /E <entry> /Fo specialise_<entry>.dxbc specialise.hlsl
//   dxc -T ps_6_0 -E <entry> -Fo specialise_<entry>.dxil specialise.hlsl
// `word`: a material word decides whether a lane of a vertex colour is read, as a select (fxc's movc on the word's row
// of the constant buffer). `integer`: a signed word compared with a number decides which lanes are read. `root`: a word's
// square root decides which lanes are read (a function the specialiser does not evaluate).
// (fxc 10.1 and dxc 1.7.2308.16, of the Windows 10 SDK 10.0.26100.0)
cbuffer Material : register(b0) {
    bool weighted;
    float scale;
    int mode;
};
float4 word(float4 position : SV_Position, float4 colour : COLOR0) : SV_Target0 {
    float mask = colour.x;
    if (weighted) {
        mask *= colour.w;
    }
    return float4(mask, colour.y * scale, 0, 1);
}
float4 integer(float4 position : SV_Position, float4 colour : COLOR0) : SV_Target0 {
    return mode > 2 ? colour : colour.wzyx;
}
float4 root(float4 position : SV_Position, float4 colour : COLOR0) : SV_Target0 {
    return sqrt(scale) > 1.5 ? colour : colour.wzyx;
}
