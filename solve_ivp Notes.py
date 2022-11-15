'''
All code taken from
https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
'''

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def func(t, y): 
    return -0.5 * y

sol = solve_ivp(func, [0, 10], [2, 4], t_eval=np.linspace(0, 10, 1024))

plt.plot(sol.t, sol.y[0], "rx")
plt.plot(sol.t, sol.y[1], "kx")

'''
Integrates the given function from t0 to tf multiple times, for initial
conditions init1 and init 2, at all specified tvals.

Then plots the time points (sol.t) against each solution (sol.y[0], sol.y[1])
'''

#%%
# Cannon firing a ball upwards

def upward_cannon(t, y): 
    return [y[1], -0.5]
def hit_ground(t, y): 
    return y[0]
hit_ground.terminal = True
hit_ground.direction = -1
sol = solve_ivp(upward_cannon, [0, 100], [0, 10], t_eval = np.linspace(0, 100, 100),
                events=hit_ground, dense_output=True)

plt.plot(sol.t, sol.y[0], "rx")

'''
"events" are a function/event that the solver tracks and keeps watch on.
event must return a float.
The solver finds values of t for which the function "event(t,y)" = 0
    
    event.terminal = True

This terminates the integration when the event conditions are met

    event.direction

If -1, then going from POSITIVE TO NEGATIVE fires the event. +1 is vice versa.
If 0, then either direction causes the event to fire

So, here, the solver finds when hit_ground = 0, i.e when the y-position y[0] = 0.
'''

#%%
# Multiple events

def upward_cannon(t, y): 
    return [y[1], -0.5] # acceleration = -0.5 length/time step
def hit_ground(t, y): 
    return y[0]
def apex(t, y):
    return y[1]
hit_ground.terminal = True
hit_ground.direction = -1
sol = solve_ivp(upward_cannon, [0, 100], [0, 10], events=(hit_ground, apex),
                dense_output=True)

plt.plot(sol.t, sol.y[0], "rx")

'''
Now including another event, which finds where the ball reaches its apex

sol.t_events returns the times the events fired

To find the position where the event fired, we access it with the sol.sol object
This is only accessible with dense_output = True

sol.sol(sol.t_events[1][0]) finds the y-position of the apex

sol.sol(sol.t_events[1][0]+1) finds the y-position 1 timestep after the apex
'''

#%%
# Bouncing ball fired upwards
def upward_cannon(t, y):
    if y[0] < 0:
        y[1] = -0.8*y[1]
    else:
        pass
    return [y[1], -0.5]

sol = solve_ivp(upward_cannon, [0, 100], [0, 10], t_eval=np.linspace(0, 100, 1024))
plt.plot(sol.t, sol.y[0])

#%%
# Additional parameters in the system

def lotkavolterra(t, z, a, b, c, d):
    x, y = z
    dx = a*x - b*x*y
    dy = -c*y + d*x*y
    return [dx, dy]

sol = solve_ivp(lotkavolterra, [0, 15], [10, 5], args=(1.5, 1, 3, 1),
                dense_output=True)

t = np.linspace(0, 15, 300)
z = sol.sol(t)
import matplotlib.pyplot as plt
plt.plot(t, z.T)
plt.xlabel('t')
plt.legend(['x', 'y'], shadow=True)
plt.title('Lotka-Volterra System')
plt.show()
