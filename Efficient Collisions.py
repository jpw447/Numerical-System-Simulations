'''
One way to solve collisions (such as a ball bouncing off a floor) is to include
an if statement as in the first cell.

Alternatively, we can define a function that will solve the system for every
collision we get, by terminating the integration at each colision, reflecting
the relevant velocities, and then continuing the integration from that point.

The idea of this is that the solver doesn't have to check the conditions every
single time, and instead just looks for a sign change.

Larger accelerations can cause the solver to take longer to run (compare g=0.5 to 9.81)
'''
#%%
import numpy as np
import matplotlit.pyplot as plt
from scipy.integrate import solve_ivp
import time

def bouncing_ball(t, y, coeff_rest, g):
    if y[0] < 0:
        y[1] = -coeff_rest*y[1]
    else:
        pass
    return [y[1], g]

g = -9.81
coeff_rest = 0.8

# In order [y(0), ydot(0)]
initial_conditions = [0, 50]
tf = 100
tvals = np.linspace(0, tf, 1024)

start = time.time()
sol = solve_ivp(bouncing_ball, [0, tf], initial_conditions, t_eval=tvals, 
                args=(coeff_rest, g))
duration = time.time() - start
print("Solution took {:.3f} seconds to calculate".format(duration))

plt.plot(sol.t, sol.y[0])

#%%
# https://scipython.com/blog/chaotic-balls/