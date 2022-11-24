import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def newtonian_gravity(t, y, constants):
    x, vx, y, vy = y
    G, M = constants
    prefactor = G*M
    r = (x**2 + y**2)**0.5
    
    xdot = vx
    xddot = -prefactor*x/r**3
    ydot = vy
    yddot = -prefactor*y/r**3
    return [xdot, xddot, ydot, yddot]

M_sun = 2e30
G = 6.67e-11
AU = 1.4959e11


vy_init = np.sqrt(G*M_sun/AU)
ic = [AU, 0, 0, vy_init]

num_orbits = 4
t0 = 0
tf = 24*365*60*60 * num_orbits

sol = solve_ivp(newtonian_gravity, [t0, tf], ic, args=([G, M_sun],), dense_output=True)
tvals = np.linspace(0, tf, 2048)
soln = sol.sol(tvals)
x = soln[0]
vx = soln[1]
y = soln[2]
vy = soln[3]

plt.figure()
plt.plot(x, y)

KE = 0.5*(vx*vx+vy*vy)
plt.figure()
plt.plot(tvals, KE)

'''
Issues:
    Energy is not conserved. Check differential equations/rewrite
'''