'''
One way to solve collisions (such as a ball bouncing off a floor) is to include
an if statement as in the first cell.

Alternatively, we can define a function that will solve the system for every
collision we get, by terminating the integration at each colision, reflecting
the relevant velocities, and then continuing the integration from that point.

The idea of this is that the solver doesn't have to check the conditions every
single time, and instead just looks for a sign change.

Larger accelerations can cause the solver to take longer to run (compare g=0.5 to 9.81)

Using events, a current bug exists where the ball "oscillates" of sorts around
y=0 before phasing through the ground.
'''
#%%
import numpy as np
import matplotlib.pyplot as plt
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
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def bouncing_ball(t, y, g):
    return [y[1], g]

def event(t, y, g):
    return y[0]
event.terminal = True
event.terminal = -1

g = -9.81

# In order [y(0), ydot(0)]
initial_conditions = [0.1, 50]
tf = 20
tvals = np.linspace(0, tf, 64)

sol = solve_ivp(bouncing_ball, [0, tf], initial_conditions, 
                events=event, args=(g,), dense_output=True)

# Solves the system for the provided tvals, regardless of events
soln = sol.sol(tvals)
fig1, ax1 = plt.subplots(figsize=(8,6))
ax1.plot(tvals, soln[0])
ax1.set_title("Solution from $t_{0}$ to $t_{f}$, ignoring events", fontsize=16)

# Solves the system, including events
tvals = np.linspace(0, sol.t[-1], 100)
soln = sol.sol(tvals)
fig2, ax2 = plt.subplots(figsize=(8,6))
ax2.plot(sol.t, sol.y[0], "k", label="Scipy sol object")
ax2.plot(tvals, soln[0], "r", label="Solved for $t_{0} \\rightarrow t_{event}$")
ax2.set_title("Solution from $t_{0}$ to $t_{event}$", fontsize=16)
ax2.legend(fontsize=16)

#%%
# Multiple bounces using events
# Current error: if run for a sufficiently long period of time, the ball will
# phase through the floor
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

# Functions to solve the system and check for bounces
def bouncing_ball(t, y, g):
    return [y[1], g]

def event(t, y, g):
    return y[0]
event.terminal = True
event.direction = -1

# Parameters
initial_conditions = [0.1, 50]
t0 = 0
tf = 200
g = -9.81
coeff_rest = 0.9

# Lists for the solution 
y = []
t = []

# Looping until tf is reached
start = time.time()
while True:
    soln = solve_ivp(bouncing_ball, [t0, tf], initial_conditions,
                     events=event, args=(g,), dense_output=True)
    
    # Calculating solution up until end of integration (bounce or tf)  
    t_end = soln.t[-1]
    num_timesteps = 2048
    
    tvals = np.linspace(t0, t_end, num_timesteps)
    new_sol = soln.sol(tvals)
    y.append(new_sol[0])
    t.append(tvals)
        
    if soln.status == 1:
        # If a bounce occurred, make new initial conditions and range
        # of t for repeat
        initial_conditions = (new_sol[0][-1], -coeff_rest*new_sol[1][-1])
        t0 = t_end
    else:
        # When we reach tf, break loop
        break
    
y = np.concatenate(y)
t = np.concatenate(t)
duration = time.time() - start
print("Solution took "+str(duration)+" seconds")
print(len(t))

fig, ax = plt.subplots(figsize=(8,6))
ax.set_title("Bouncing Ball", fontsize=20)
ax.set_xlabel("$t$", fontsize=16)
ax.set_ylabel("$y$", fontsize=16)
ax.plot(t,y)