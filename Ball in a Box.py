'''
Code to simulate a particle moving in free space, constrained to be within
a box
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def free_particle(t, y):
    return [y[1], 0, y[3], 0]

def vertical_impact(t, y):
    return 1-y[0]
vertical_impact.terminal = True

def horizontal_impact(t, y):
    return 1-y[2]
horizontal_impact.terminal=True

# [x, xdot, y, ydot]
initial_conditions = [0, 1, 0, 0.5]
t0 = 0
tf = 2
x = []
y = []
t = []

'''
Issues:
    How do we check which event fired, and which velocity to invert?
    Takes forever to solve for non-zero vy.
        Maybe try to start more simply and have a floor+ceiling to bounce the
        ball?
'''

while True:
    sol = solve_ivp(free_particle, [t0, tf], initial_conditions, 
                events=(vertical_impact, horizontal_impact), dense_output=True)
    t_end = sol.t[-1]
    tvals = np.linspace(0, t_end, 1024)
    
    new_soln = sol.sol(tvals)
    x.append(new_soln[0])
    y.append(new_soln[2])
    t.append(tvals)
    
    # If integration was interrupted
    if sol.status == 1:
        t0 = t_end
        new_x = new_soln[0][-1]
        new_vx = new_soln[1][-1]
        new_y = new_soln[2][-1]
        new_vy = new_soln[3][-1]
        initial_conditions = [new_x, -new_vx, new_y, new_vy]
    else:
        break

x = np.concatenate(x)
y = np.concatenate(y)
t = np.concatenate(t)
plt.plot(x, y)