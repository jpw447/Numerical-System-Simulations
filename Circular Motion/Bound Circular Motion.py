'''
Code to simulate a body undergoing circular motion, whilst being constrained by
some force (e.g. a rope providing tension).

Since the angular velocity omega (or w here) is not given a direction in this
code, the angle only ever increases and does not follow the expected
sinusoidal behaviour.

Including np.sign makes the code run significantly slower. The best solution to
get the sinusoidal variation in theta, that I can think of, is to use
Cartesian co-ordinates, which is in the second cell.
'''
# POLAR CO-ORDINATES


def polar_soln(t, parameters):
    r, v_r, v_t, w, theta = parameters

    # Bound circular motion
    dr = v_r
    dv_r = 0
    dv_t = 0
    dw = 0
    dtheta = w

    return [dr, dv_r, dv_t, dw, dtheta]


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp

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
    soln = solve_ivp(polar_soln, [0, 2*np.pi], ic, method='RK23',
                     dense_output=True, rtol=1e-8)

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

#%%
# CARTESIAN CO-ORDINATES


def cart_soln(t, parameters):
    x, y, vx, vy = parameters
    vt2 = vx**2 + vy**2
    r2 = x**2 + y**2

    # Bound circular motion in Cartesian co-ordinates
    dx = vx
    dy = vy
    dvx = -vt2*x/r2
    dvy = -vt2*y/r2

    return [dx, dy, dvx, dvy]


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp

    # Parameters and initial conditions
    m = 1
    E = 2*np.pi**2  # Chosen to make omega = 2pi
    x_init = 1
    y_init = 0
    vx_init = 0
    r_init = (x_init**2 + y_init**2)**0.5
    w_init = np.sqrt(2*E/(m*r_init**2))
    vy_init = w_init * (x_init**2 + y_init**2)**0.5

    ic = [x_init, y_init, vx_init, vy_init]

    # Solution from 0 to t=2pi
    soln = solve_ivp(cart_soln, [0, 2*np.pi], ic, method='RK45',
                     dense_output=True, rtol=1e-8)

    tvals = np.linspace(0, 2*np.pi, 2048)
    soln = soln.sol(tvals)

    # Variables from solution and energy variation
    x = soln[0]
    y = soln[1]
    vx = soln[2]
    vy = soln[3]

    r = np.hypot(x, y)
    # Defining theta piece-wise to avoid arccos restricting theta to be
    # 0<theta<pi
    theta = np.zeros(len(tvals))
    theta = np.where(y >= 0, np.arccos(x/r), 0)
    theta = np.where(y <= 0, -np.arccos(x/r), theta)

    # Projections of trajectory
    # Cartesian
    fig_proj = plt.figure(figsize=(10, 5))
    ax_cart = plt.subplot(121)
    ax_cart.plot(x, y)
    ax_cart.set_xlabel("$x$", fontsize=12)
    ax_cart.set_ylabel("$y$", fontsize=12)
    ax_cart.axis("equal")
    # Polar
    ax_polar = plt.subplot(122, projection="polar")
    ax_polar.plot(theta, r)
    ax_polar.set_ylim(0, 1.1*np.max(r))

    # Time-dependence of different variables
    fig_time, ax_time = plt.subplots(1, 2, figsize=(10, 5))
    ax_time[0].plot(tvals, x)
    ax_time[0].set_xlabel("$t$", fontsize=12)
    ax_time[0].set_ylabel("$x$", fontsize=12)
    ax_time[0].set_title("$x(t)$", fontsize=14)
    ax_time[1].plot(tvals, y)
    ax_time[1].set_xlabel("$t$", fontsize=12)
    ax_time[1].set_ylabel("$y$", fontsize=12)
    ax_time[1].set_title("$y(t)$", fontsize=14)
