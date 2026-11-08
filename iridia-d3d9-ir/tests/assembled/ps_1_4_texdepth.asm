; ps_1_4 texdepth: the depth r5.r / r5.g, r5.g taken as 1 past 1, and the depth 1 where r5.g is 0 (the instruction
; reference: "If r5.g = 0, the result of r5.r / r5.g = 1.0")
; run: TEXCOORD0=0.25,0.5,0,1 c0=0,0,0,1 -> SV_Target0=0,0,0,1 SV_Depth0=0.5
; run: TEXCOORD0=0.25,2,0,1 c0=0,0,0,1 -> SV_Target0=0,0,0,1 SV_Depth0=0.25
; run: TEXCOORD0=0.25,0,0,1 c0=0,0,0,1 -> SV_Target0=0,0,0,1 SV_Depth0=1
; run: TEXCOORD0=0.25,-0,0,1 c0=0,0,0,1 -> SV_Target0=0,0,0,1 SV_Depth0=1
; run: TEXCOORD0=0,0,0,1 c0=0,0,0,1 -> SV_Target0=0,0,0,1 SV_Depth0=1
ps_1_4
texcrd r5.rgb, t0
phase
texdepth r5
mov r0, c0
