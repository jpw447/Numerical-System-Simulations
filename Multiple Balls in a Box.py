'''
Code to simulate two particles moving in free space, constrained to be within
a box.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def free_particle(t, y):
    return [y[2], y[3], 0, 0, y[5], y[6], 0, 0]


def wall_left_1(t, y):
    return -2-y[0]
wall_left_1.terminal = True


def wall_right_1(t, y):
    return 2-y[0]
wall_right_1.terminal = True


def wall_left_2(t, y):
    return -2-y[4]
wall_left_2.terminal = True


def wall_right_2(t,y):
    return 2-y[4]
wall_right_2.terminal = True


def collision(t, y):
    return np.hypot(y[0], y[1]) - np.hypot(y[4], y[5])

collision.terminal = True

# [x, xdot, y, ydot]
ic_1 = [0, 0, 1, 1]
ic_2 = [1, 1, -1, -1]
initial_conditions = ic_1 + ic_2

# Try writing as a 2D array and using ravel() to flatten to 1D. Then
# reshape it for calculations?

t0 = 0
tf = 10
x1 = []
y1 = []
x2 = []
y2 = []
t = []
size = 1
coeff = 1  # coefficient of restitution for wall collisions

'''
Notes:
    The number of events can be easily reduced by making a polynomial whose
    roots are all of the conditions. The number of checks later would be the
    same, however

Issues:
    Keeps getting stuck, and the collision check isn't very good. If the
    particles are at (2,2) and (-2,-2), the code will think they've collided
'''

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
    y1_impact = new_sol[1][-1]
    x2_impact = new_sol[4][-1]
    y2_impact = new_sol[5][-1]
    # print("Termination")

    x1.append(new_sol[0])
    y1.append(new_sol[1])
    x2.append(new_sol[4])
    y2.append(new_sol[5])
    t.append(tvals)

    if soln.status == 1:
        # If intergration was interrupted (i.e. there was an impact)
        vx1 = new_sol[2][-1]
        vy1 = new_sol[3][-1]
        vx2 = new_sol[6][-1]
        vy2 = new_sol[7][-1]

        r1 = np.round(np.array([x1_impact, y1_impact]), 6)
        r2 = np.round(np.array([x2_impact, y2_impact]), 6)

        bump_x1, bump_y1, bump_x2, bump_y2 = 0, 0, 0, 0

        # Used to separate particles after collision
        if (r1 == r2).all():
            print("Collision!")
            vx1 = -vx1
            vy1 = -vy1
            vx2 = -vx2
            vy2 = -vy2
            if r1[0] > r2[0]:
                bump_x1 = 1e-10
                bump_x2 = -1e-10
            else:
                bump_x1 = -1e-10
                bump_x2 = 1e-10
            if r1[1] > r2[1]:
                bump_y1 = 1e-10
                bump_y2 = -1e-10
            else:
                bump_y1 = -1e-10
                bump_y2 = 1e-10

        # Checking positions on wall collision
        # Particle 1
        if np.round(x1_impact, 6) == 2:
            print("Right wall 1")
            bump_x1 = -1e-10
            vx1 = -vx1
        elif np.round(x1_impact, 6) == -2:
            print("Left wall 1")
            bump_x1 = 1e-10
            vx1 = -vx1

        if np.round(y1_impact, 6) == 2:
            print("Ceiling 1")
            bump_y1 = -1e-10
            vy1 = -vy1
        elif np.round(y1_impact, 6) == -2:
            print("FLoor 1")
            bump_y1 = 1e-10
            vy1 = -vy1

        # Particle 2
        if np.round(x2_impact, 6) == 2:
            print("Right wall 2")
            bump_x2 = -1e-10
            vx2 = -vx2
        elif np.round(x2_impact, 6) == -2:
            print("Left wall 2")
            bump_x2 = 1e-10
            vx2 = -vx2

        if np.round(y2_impact, 6) == 2:
            print("Ceiling 2")
            bump_y2 = -1e-10
            vy2 = -vy2
        elif np.round(y2_impact, 6) == -2:
            print("Floor 2")
            bump_y2 = 1e-10
            vy2 = -vy2

        # Update parameters for new integration
        ic_1 = [x1_impact + bump_x1, y1_impact + bump_y1, vx1, vy1]
        ic_2 = [x2_impact + bump_x2, y2_impact + bump_y2, vx2, vy2]
        initial_conditions = ic_1 + ic_2
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
