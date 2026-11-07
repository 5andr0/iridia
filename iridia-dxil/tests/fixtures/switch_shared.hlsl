// A switch several of whose cases go straight to the block after it, which takes values from the switch's block: that
// block's phis have an entry for every edge from the switch's block, the same value on each (LLVM's rule), and each
// edge carries those values. Compiled with
//   dxc -T ps_6_0 -E main (iridia-dxil/tests/fixtures/switch_shared.dxil)
// For s = 0, 2 or 5: (7, a * 2); for s = 1: (a * 3, 4); for any other s: (a - 2, a * 2).
float4 main(nointerpolation uint s : S, float a : A) : SV_Target
{
    float r = 7.0;
    float q = a * 2.0;
    switch (s)
    {
    case 0:
    case 2:
    case 5:
        break;
    case 1:
        r = a * 3.0;
        q = 4.0;
        break;
    default:
        r = a - 2.0;
        break;
    }
    return float4(r, q, 0.0, 1.0);
}
