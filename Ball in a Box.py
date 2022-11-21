'''
Code to simulate a particle moving in free space, constrained to be within
two walls.

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
while True:
    soln = solve_ivp(free_particle, [t0, tf], initial_conditions, args=(thickness,),
                     events=(right_wall, left_wall), dense_output=True)
    
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

#%%
'''
Now constraining the particle to be within a box (4 events).

If statements used to check the particle position after integration gets 
interrupted are unlikely to slow the code down, as there are significantly
fewer of these than if they were in the solving function free_particle()

Could reduce this to 1 event by creating an array giving the box position and
compare it to the x, y values? 

|box - x| + |box - y| > threshold value
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def free_particle(t, y, size):
    return [y[1], 0, y[3], 0]

def right_wall(t, y, size):
    return size-y[0]

def left_wall(t, y, size):
    return -size-y[0]

def ceiling(t, y, size):
    return size-y[2]

def floor(t, y, size):
    return -size-y[2]

right_wall.terminal = True
left_wall.terminal = True
ceiling.terminal = True
floor.terminal = True

# [x, xdot, y, ydot]

initial_conditions = [0, 1, 0, 0.5]
t0 = 0
tf = 10
x = []
y = []
vx_arr = []
vy_arr = []
t = []
size = 1

while True:
    soln = solve_ivp(free_particle, [t0, tf], initial_conditions, args=(size,),
                     events=(right_wall, left_wall, ceiling, floor),
                     dense_output=True)
    
    # Calculating solution up until end of integration (bounce or tf)  
    t_end = soln.t[-1]
    
    num_timesteps = 2048
    tvals = np.linspace(t0, t_end, num_timesteps)
    
    new_sol = soln.sol(tvals)
    x_impact = new_sol[0][-1]
    y_impact = new_sol[2][-1]

    x.append(new_sol[0])
    y.append(new_sol[2])
    vx_arr.append(new_sol[1])
    vy_arr.append(new_sol[3])
    t.append(tvals)
        
    if soln.status == 1:
        # If intergration was interrupte (i.e. there was an impact)
        vx = new_sol[1][-1]
        vy = new_sol[3][-1]
        
        # Used to kick the particle away from the wall
        if vx > 0:
            bump_x = -1e-10
        else:
            bump_x = 1e-10
            
        if vy > 0:
            bump_y = -1e-10
        else:
            bump_y = 1e-10
            
        # Checking which velocity needs inverting by manually checking collision
        if abs(np.round(x_impact, 6)) == 1:
            vx = -vx
        elif abs(np.round(y_impact, 6)) == 1:
            vy = -vy
        else:
            pass
        
        # Update parameters for new integration
        initial_conditions = (x_impact + bump_x, vx, y_impact + bump_y, vy)
        t0 = t_end
    else:
        # When we reach tf, break loop
        break

# Lists to numpy arrays and plot
x = np.concatenate(x)
y = np.concatenate(y)
t = np.concatenate(t)
vx_arr = np.concatenate(vx_arr)
vy_arr = np.concatenate(vy_arr)

# Energy calculations
vx_mag = vx_arr * vx_arr
vy_mag = vy_arr * vy_arr
KE = 0.5*(vx_mag + vy_mag)
KE_init = 0.5*(initial_conditions[1]**2 + initial_conditions[3]**2)
delta_E = (KE - KE_init)/KE_init

fig, ax = plt.subplots()
ax.plot(x, y, "r--")
ax.plot(0, 0, "ko", markersize=2)
for val in [-1, 1]:
    ax.hlines(val, -1, 1, colors="k")
    ax.vlines(val, -1, 1, colors="k")
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel("$x$", fontsize=16)
ax.set_ylabel("$y$", fontsize=16)

fig_energy, ax_energy = plt.subplots()
ax_energy.plot(t, delta_E)

#%%
'''
Animating the result. Code from: https://pythonforundergradengineers.com/live-plotting-with-matplotlib.html
'''
from matplotlib.animation import FuncAnimation

fig_ani, ax_ani = plt.subplots()
x_arr = x[::64]
y_arr = y[::64]

fig, ax = plt.subplots()

x_anim = []
y_anim = []

def animate(i):
    x = x_arr[i-1:i]
    y = y_arr[i-1:i]
    x_anim.append(x)
    y_anim.append(y)

    ax.clear()
    ax.plot(x, y, "ro", markersize=3)
    for val in [-1, 1]:
        ax.hlines(val, -1, 1, colors="k")
        ax.vlines(val, -1, 1, colors="k")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("$x$", fontsize=16)
    ax.set_ylabel("$y$", fontsize=16)
    
ani = FuncAnimation(fig, animate, frames=len(x_arr), interval=1/120, repeat=False)
plt.show()