; ps_3_0 dsx and dsy of an input, written to two render targets
; unrun: a derivative needs the invocation's quad, which a single run does not have
ps_3_0
dcl_texcoord0 v0
dsx r0, v0
dsy r1, v0
mov oC0, r0
mov oC1, r1
