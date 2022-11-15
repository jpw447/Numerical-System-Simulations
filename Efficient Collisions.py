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
# Acceleration due to gravity, m.s-2 (downwards!).
g = 9.81
# Time step, s.
dt = 0.001

def solve(u0):
    """Solve the equation of motion for a ball bouncing in a circle.

    u0 = [x0, vx0, y0, vy0] are the initial conditions (position and velocity).

    """

    # Initial time, final time, s.
    t0, tf = 0, 10

    def fun(t, u):
        """Return the derivatives of the dynamics variables packed into u."""
        y, ydot = u
        yddot = -g
        return ydot, yddot

    def event(t, u):
        """If the ball hits the wall of the circle, trigger the event."""
        return u[0]
    # Make sure the event terminates the integration.
    event.terminal = True

    # Keep track of the ball's position in these lists.
    y = []
    t = []
    while True:
        # Solve the equations until the ball hits the floor or until tf
        soln = solve_ivp(fun, (t0, tf), u0, events=event, dense_output=True)
        if soln.status == 1:
            # We hit the wall: save the path so far...
            tend = soln.t_events[0][0]
            # nt = int(tend - t0 / dt) + 1
            tgrid = np.linspace(t0, tend, 100)
            sol = soln.sol(tgrid)
            y.append(sol[0])
            t.append(tgrid)

            # ...and restart the integration with the reflected velocities as
            # the initial conditions.
            u0 = soln.y[0][-1], -1*soln.y[1][-1]
            t0 = soln.t[-1]
        else:
            # We're done up to tf (or, rather, the last bounce before tf).
            break
    # Concatenate all the paths between bounces together.
    return np.concatenate(y), np.concatenate(t)
    
u0 = [10, 0]
y, t = solve(u0)
    
    
    
    
    