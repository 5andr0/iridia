; ps_2_0 exp (2^x) and log (log2 |x|), each of a scalar replicated; log of a zero is -FLT_MAX, either sign (the instruction
; reference's code fragment)
; run: c0=3,8,-0.5,-8 -> SV_Target0=8,8,8,8 SV_Target1=3,3,3,3
; run: c0=3,8,-0.5,0 -> SV_Target0=8,8,8,8 SV_Target1=0xff7fffff,0xff7fffff,0xff7fffff,0xff7fffff
; run: c0=3,8,-0.5,-0 -> SV_Target0=8,8,8,8 SV_Target1=0xff7fffff,0xff7fffff,0xff7fffff,0xff7fffff
ps_2_0
mov r0, c0
exp r1, r0.x
log r2, r0.w
mov oC0, r1
mov oC1, r2
