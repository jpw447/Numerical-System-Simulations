'''
Code to simulate two particles moving in free space, constrained to be within
a box.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def free_particle(t, y):
    return [y[1], 0, y[3], 0]

# def right_wall(t, y, size):
#     return size-y[0]

# def left_wall(t, y, size):
#     return -size-y[0]

# def ceiling_1(t, y, size):
#     return size-y[0]

# def floor(t, y, size):
#     return -size-y[2]

def wall_left_1(t, y):
    return -2-y[0]
wall_left_1.terminal = True

def wall_right_1(t, y):
    return 2-y[0]
wall_right_1.terminal = True

def wall_left_2(t, y):
    return -2-y[2]
wall_left_2.terminal = True

def wall_right_2(t,y):
    return 2-y[2]
wall_right_2.terminal = True

def collision(t, y):
    return y[0] - y[2]

collision.terminal = True

# [x, xdot, y, ydot]

initial_conditions = [0, 1, 1, -1]

# Try writing as a 2D array and using ravel() to flatten to 1D. Then
# reshape it for calculations?
                  
t0 = 0
tf = 100
x1 = []
x2 = []
t = []
size = 1
coeff = 1 # coefficient of restitution for wall collisions


while True:
    soln = solve_ivp(free_particle, [t0, tf], initial_conditions,
                     events=(collision, wall_left_1, wall_right_1, wall_left_2,
                             wall_right_2),
                     dense_output=True)
    
    # Calculating solution up until end of integration (bounce or tf)  
    t_end = soln.t[-1]
    
    num_timesteps = 2048
    tvals = np.linspace(t0, t_end, num_timesteps)
    
    new_sol = soln.sol(tvals)
    x1_impact = new_sol[0][-1]
    x2_impact = new_sol[2][-1]
    
    x1.append(new_sol[0])
    x2.append(new_sol[2])
    t.append(tvals)
        
    if soln.status == 1:
        # If intergration was interrupte (i.e. there was an impact)
        vx1 = new_sol[1][-1]
        vx2 = new_sol[3][-1]
            
        # Used to separate particles after collision
        if vx1 > 0:
            bump_x1 = -1e-10
        else:
            bump_x1 = 1e-10
            
        if vx2 > 0:
            bump_x2 = -1e-10
        else:
            bump_x2 = 1e-10
            
        # Checking positions on wall collision
        if np.round(x1_impact,6) == 2:
            bump_x1 = -1e-10
            vx1 = -coeff*vx1
        elif np.round(x1_impact, 6) == -2:
            bump_x1 = 1e-10
            vx1 = -coeff*vx1
            
        if np.round(x2_impact,6) == 2:
            bump_x2 = -1e-10
            vx2 = -coeff*vx2
        elif np.round(x2_impact, 6) == -2:
            bump_x2 = 1e-10
            vx2 = -coeff*vx2
            
        # Checking which velocity needs inverting by manually checking collision
        if abs(np.round(x1_impact-x2_impact, 6)) == 0:
            vx1 = -vx1
            vx2 = -vx2
        
        # Update parameters for new integration
        initial_conditions = (x1_impact + bump_x1, vx1, x2_impact + bump_x2, vx2)
        t0 = t_end
    else:
        # When we reach tf, break loop
        break

# Lists to numpy arrays and plot
x1 = np.concatenate(x1)
x2 = np.concatenate(x2)
t = np.concatenate(t)

fig, ax = plt.subplots()
ax.plot(x1, t, "r")
ax.plot(x2, t, "k")
ax.set_xlabel("$x$", fontsize=16)
ax.set_ylabel("$t$", fontsize=16)