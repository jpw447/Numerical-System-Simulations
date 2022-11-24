'''
Code to simulate two particles moving in free space, constrained to be within
a box.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def free_particle(t, y):
    return [y[2], y[3], 0, 0, y[6], y[7], 0, 0]


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


def ceiling_1(t,y):
    return 2-y[1]
ceiling_1.terminal = True


def ceiling_2(t,y):
    return 2-y[5]
ceiling_2.terminal = True


def floor_1(t,y):
    return -2-y[1]
floor_1.terminal = True


def floor_2(t,y):
    return -2-y[5]
floor_2.terminal = True

def collision(t, y):
    return (y[0] - y[4])**2 + (y[1] - y[5])**2
collision.terminal = True

# [x, xdot, y, ydot]
ic_1 = [0, 0, 0.67, 1]
ic_2 = [1, 1, -0.5, -1]
initial_conditions = ic_1 + ic_2

# Try writing as a 2D array and using ravel() to flatten to 1D. Then
# reshape it for calculations?

t0 = 0
tf = 20
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
    Integration terminates without giving a reason when there's a "collision",
    which was when x1**2 + y1**2 = x2**2 + y2**2. Need a way to identify when
    the co-ordinates are the same using a scalar equation
'''


while True:
    soln = solve_ivp(free_particle, [t0, tf], initial_conditions,
                     events=(collision, wall_left_1, wall_right_1, wall_left_2,
                             wall_right_2, ceiling_1, ceiling_2, floor_1, floor_2),
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

        distance = (x1_impact - x2_impact)**2 + (y1_impact - y2_impact)**2

        bump_x1, bump_y1, bump_x2, bump_y2 = 0, 0, 0, 0

        # Checking if particles have collided
        if distance == 0:
            vx1 = -vx1
            vy1 = -vy1
            vx2 = -vx2
            vy2 = -vy2
            if x1_impact > x2_impact:
                bump_x1 = 1e-10
                bump_x2 = -1e-10
            else:
                bump_x1 = -1e-10
                bump_x2 = 1e-10
            if y1_impact > y2_impact:
                bump_y1 = 1e-10
                bump_y2 = -1e-10
            else:
                bump_y1 = -1e-10
                bump_y2 = 1e-10

        # Checking positions on wall collision
        # Particle 1 walls
        if np.round(x1_impact, 6) == 2:
            bump_x1 = -1e-10
            vx1 = -vx1
        elif np.round(x1_impact, 6) == -2:
            bump_x1 = 1e-10
            vx1 = -vx1

        # Particle 1 ceiling and floor
        if np.round(y1_impact, 6) == 2:
            bump_y1 = -1e-10
            vy1 = -vy1
        elif np.round(y1_impact, 6) == -2:
            bump_y1 = 1e-10
            vy1 = -vy1

        # Particle 2 walls
        if np.round(x2_impact, 6) == 2:
            bump_x2 = -1e-10
            vx2 = -vx2
        elif np.round(x2_impact, 6) == -2:
            bump_x2 = 1e-10
            vx2 = -vx2

        # Particle 2 ceiling and floor
        if np.round(y2_impact, 6) == 2:
            bump_y2 = -1e-10
            vy2 = -vy2
        elif np.round(y2_impact, 6) == -2:
            bump_y2 = 1e-10
            vy2 = -vy2

        # Update parameters for new integration
        ic_1 = [x1_impact + bump_x1, y1_impact + bump_y1, vx1, vy1]
        ic_2 = [x2_impact + bump_x2, y2_impact + bump_y2, vx2, vy2]
        initial_conditions = ic_1 + ic_2
        t0 = t_end
    else:
        # When we reach tf, break loop
        print("Done!")
        break

# Lists to numpy arrays and plot
x1 = np.concatenate(x1)
y1 = np.concatenate(y1)
x2 = np.concatenate(x2)
y2 = np.concatenate(y2)
t = np.concatenate(t)

fig_box, ax_box = plt.subplots()
ax_box.plot(x1, y1, "r", label="Particle 1")
ax_box.plot(x2, y2, "b", label="Particle 2")
ax_box.plot(0, 0, "ro", markersize=6)
ax_box.plot(1, 1, "bo", markersize=6)
for val in [-2, 2]:
    ax_box.hlines(val, -2, 2, "k")
    ax_box.vlines(val, -2, 2, "k")
ax_box.set_xlabel("$x$", fontsize=16)
ax_box.set_ylabel("$y$", fontsize=16)
ax_box.set_title("Multiple Particles within a Box", fontsize=20)
ax_box.set_aspect("equal")
ax_box.legend(fontsize=12)

fig_world, ax_world = plt.subplots()
ax_world.plot(x1, t, "r", label="Particle 1")
ax_world.plot(x2, t, "b", label="Particle 2")
ax_world.set_xlabel("$t$", fontsize=16)
ax_world.set_ylabel("$x$", fontsize=16)
ax_world.set_title("Worldline Plots", fontsize=20)
ax_world.legend(fontsize=12)
