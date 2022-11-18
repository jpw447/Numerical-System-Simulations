'''
Code to simulate a particle moving in free space, constrained to be within
a box.

soln.t_events returns a 2D array. The first element returns the event that fired,
in the same order they were provided to solve_ivp
The second element is the time at which it was fired
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def free_particle(t, y, thickness):
    return [y[1], 0]

def right_wall(t, y, thickness):
    return thickness-y[0]

def left_wall(t, y, thickness):
    return -thickness-y[0]
right_wall.terminal = True
left_wall.terminal = True

# [x, xdot]
initial_conditions = [0, 1]
t0 = 0
tf = 10
x = []
t = []
thickness = 1

'''
Issues:
'''
for i in range(100):
    soln = solve_ivp(free_particle, [t0, tf], initial_conditions, args=(thickness,),
                     events=(right_wall, left_wall), dense_output=True)
    t_impact = soln.t_events[0]
    
    # Calculating solution up until end of integration (bounce or tf)  
    t_end = soln.t[-1]
    
    num_timesteps = 2048
    tvals = np.linspace(t0, t_end, num_timesteps)
    
    new_sol = soln.sol(tvals)
    x_impact = new_sol[0][-1]

    x.append(new_sol[0])
    t.append(tvals)
    print("Impact x="+str(new_sol[0][-1]))
        
    if soln.status == 1:
        '''If a bounce occurred, make new initial conditions and range
        of t for repeat. Need if statement to kick ball away from the wall to 
        prevent it getting stuck.'''
        vx = new_sol[1][-1]
        if vx > 0:
            bump = -1e-10
        else:
            bump = 1e-10
        initial_conditions = (new_sol[0][-1] + bump, -new_sol[1][-1])
        t0 = t_end
    else:
        # When we reach tf, break loop
        break

x = np.concatenate(x)
t = np.concatenate(t)
fig, ax = plt.subplots()
ax.plot(x, t)
ax.set_xlabel("$x$", fontsize=16)
ax.set_ylabel("$t$", fontsize=16)