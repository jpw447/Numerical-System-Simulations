import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def newtonian_gravity(t, y, constants):
    x, vx, y, vy = y
    prefactor, R = constants
    r = (x**2 + y**2)/R
    
    xdot = vx
    xddot = -prefactor*x/r**3
    ydot = vy
    yddot = -prefactor*y/r**3
    return [xdot, xddot, ydot, yddot]

M_sun = 2e30
G = 6.67e-11
P = 365*24*60**2
AU = 1.4959e11

const = G*M_sun*P/(AU**3)

vy_init = 2*np.pi # AU/year
ic = [1, 0, 0, vy_init]

t0 = 0
tf = 1

sol = solve_ivp(newtonian_gravity, [t0, tf], ic, args=([const, AU],), dense_output=True)
tvals = np.linspace(0, 1, 1024)
soln = sol.sol(tvals)
x = soln[0]
y = soln[1]

plt.plot(tvals, y)

'''
Values are orders of magnitude off, primarily, the dynamics are incorrect
'''