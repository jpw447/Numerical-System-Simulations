'''
Code to simulate a body under a centripetal force pushing it radially outwards.
Energy seems to be conserved (particularly as method is changed from RK23 to
DOP853 and relative tolerance is changed to ~1e-8), so it's likely to be
correct
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def func(t, parameters):
    r, v_r, v_t, w, theta = parameters

    # Unbound circular motion
    '''
    dr = v_r
    dv_r = w**2 * r
    dv_t = w*r
    dw = np.sqrt(2*E/m)*dr*(-1/(r**2))
    dtheta = w
    '''
    # Bound circular motion
    dr = 0
    dv_r = 0
    dv_t = 0
    dw = 0
    dtheta = w

    return [dr, dv_r, dv_t, dw, dtheta]


# Parameters and initial conditions
m = 1
r_init = 1
E = 2*np.pi**2  # Chosen to make omega = 2pi
w_init = np.sqrt(2*E/(m*r_init**2))
v_t_init = w_init*r_init
v_r_init = 0
theta_init = 0

ic = [r_init, v_r_init, v_t_init, w_init, theta_init]

# Solution from 0 to t=2pi
soln = solve_ivp(func, [0, 2*np.pi], ic, method='RK23', dense_output=True,
                 rtol=1e-8)

tvals = np.linspace(0, 2*np.pi, 128)
soln = soln.sol(tvals)

# Variables from solution and energy variation
r = soln[0]
v_r = soln[1]
v_t = soln[2]
w = soln[3]
theta = soln[4]
delta_E = (0.5*m*r**2*w**2 - E)/E

# Polar projection of trajectory
fig_pol, ax_pol = plt.subplots(subplot_kw={'projection': 'polar'})
ax_pol.plot(theta, r)

# Time-dependence of different variables
fig_time, ax_time = plt.subplots(1, 2, figsize=(10, 5))
ax_time[0].plot(tvals, r)
ax_time[0].set_xlabel("$t$", fontsize=12)
ax_time[0].set_ylabel("$r$", fontsize=12)
ax_time[0].set_title("Radius-time dependence", fontsize=14)
ax_time[1].plot(tvals, theta)
ax_time[1].set_xlabel("$t$", fontsize=12)
ax_time[1].set_ylabel("$\\theta$", fontsize=12)
ax_time[1].set_title("Angle-time dependence", fontsize=14)
