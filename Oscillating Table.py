import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

'''
Simulating a bouncing ball on an oscillating table
'''


def oscillating_table(t, y, args):
    A, omega = args
    ydot = A*np.sin(omega*t)
    return [ydot]


y = [0]
A = 2
period = 2
omega = 2*np.pi/period
t0 = 0
tf = 2

soln = solve_ivp(oscillating_table, [t0, tf], y, args=([A, omega],),
                 dense_output=True)

tvals = np.linspace(t0, tf, 1024)
sol = soln.sol(tvals)

plt.plot(tvals, sol[0])

#%%


def bouncing_ball(t, y, args):
    g, A, omega = args
    vy_ball = y[1]
    vy_table = A*np.sin(omega*t)
    return [vy_ball, g, vy_table]


def collision(t, y, args):
    return y[0] - y[2]
collision.terminal = True
collision.direction = -1


g = -9.81
A = 2
period = 1
omega = 2*np.pi/period
parameters = [g, A, omega]
# [y_ball, vy_ball, y_table]
y = [1, 0, 0]
t0 = 0
tf = 3

ball_yvals = []
table_yvals = []
tvals = []

while True:
    soln = solve_ivp(bouncing_ball, [t0, tf], y, args=(parameters,), events=collision,
                 dense_output=True)

    t_end = soln.t[-1]
    
    if soln.status == 1:
        t = np.linspace(t0, t_end, 1024)
        sol = soln.sol(t)
        tvals.append(t)
        ball_yvals.append(sol[0])
        table_yvals.append(sol[2])
        y = [sol[0][-1], -sol[1][-1], sol[2][-1]]
        t0 = t_end
    else:
        break

tvals = np.concatenate(tvals)
ball_yvals = np.concatenate(ball_yvals)
table_yvals = np.concatenate(table_yvals)

plt.plot(tvals, ball_yvals)
plt.plot(tvals, table_yvals)