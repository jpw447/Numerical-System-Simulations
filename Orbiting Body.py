import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

'''
Will need to include the star as well. Refer to "Old N-Body.py" for the old
2-body solution from the SciComp project
'''

def field_func(t, y, const):
    x, vx, y, vy = y
    G, M = const
    r3 = (x**2 + y**2)**1.5

    xdot = vx
    xddot = -G*M*x/(r3)
    ydot = vy
    yddot = -G*M*y/(r3)

    return [xdot, xddot, ydot, yddot]


G = 6.67e-11
M_sun = 1.989e30
M_earth = 5.972e24
AU = 1.5e11
vy_init = np.sqrt(G*M_sun/AU)

t0 = 0
tf = 365*24*60**2

ic = [AU, 0, 0, vy_init]

soln = solve_ivp(field_func, [t0, tf], ic, args=([G, M_sun],), dense_output=True)

tvals = np.linspace(t0, tf, 1024)
sol = soln.sol(tvals)

x = sol[0]
vx = sol[1]
y = sol[2]
vy = sol[3]

KE = 0.5*(vx*vx + vy*vy)
PE = -G*M_sun*M_earth/(x*x + y*y)**0.5
E = KE + PE

fig, ax = plt.subplots(1, 2, figsize=(10, 6))

ax[0].plot(x, y)
ax[0].set_xlabel("$x$", fontsize=16)
ax[0].set_ylabel("$y$", fontsize=16)
ax[1].plot(tvals, KE, "g", label="KE")
ax[1].plot(tvals, PE, "r", label="PE")
ax[1].set_xlabel("$t$", fontsize=16)
ax[1].set_ylabel("Kinetic Energy", fontsize=16)
ax[1].legend(fontsize=12)
