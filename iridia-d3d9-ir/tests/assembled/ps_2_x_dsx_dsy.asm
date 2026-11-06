; ps_2_x dsx and dsy: the coordinate's derivatives along x and y
; unrun: a derivative needs the invocation's quad, which a single run does not have
ps_2_x
dcl t0
dsx r0, t0
dsy r1, t0
add r0, r0, r1
mov oC0, r0
