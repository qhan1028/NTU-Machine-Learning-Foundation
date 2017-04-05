import math as m

e = m.e

# E(u, v)
E = lambda u, v : e**u + e**(2*v) + e**(u*v) + u**2 - 2*u*v + 2*v**2 - 3*u - 2*v
# gradients
Gu = lambda u, v : e**u + v*e**(u*v) + 2*u - 2*v - 3
Gv = lambda u, v : 2*e**(2*v) + u*e**(u*v) - 2*u + 4*v - 2
Guu = lambda u, v : e**u + (v**2)*e**(u*v) + 2
Gvv = lambda u, v : 4*e**(2*v) + (u**2)*e**(u*v) + 4
Guv = lambda u, v : e**(u*v) - 2

u = v = 0
print('E(u0, v0) =', E(u, v))
for i in range(5):
	nu = -1 * Gu(u, v) / (Guu(u, v) + Guv(u, v))
	nv = -1 * Gv(u, v) / (Gvv(u, v) + Guv(u, v))
	u += nu
	v += nv
	print ('E(u', i+1, ', v', i+1, ') = ', E(u, v), sep='')
