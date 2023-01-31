'''
Here, a particle undergoes circular motion attached to an elastic rope, which
is modelled as a massless spring.

We can see from the second plot that as the radial velocity reaches a maximum,
the radius is at a minimum, where the most work has been done. The energy
change is shown in the third plot.

When the radius returns to its initial value, the radial velocity also
returns to its initial value.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumtrapz


def spring_polar_soln(t, parameters):
    r, v_r, v_t, w, E = parameters

    # Circular motion with a spring, assuming angular velocity is constant
    dr = v_r
    # Assuming spring does work on the particle
    dE = k*(r-L)*dr
    dw = dE/(m*w*r) - (w*v_r)/(2*r)
    dv_r = -(k/m)*(r-L)
    dv_t = r*dw + w*v_r

    return [dr, dv_r, dv_t, dw, dE]


if __name__ == "__main__":
    # Parameters and initial conditions
    m = 1  # Mass
    k = 1  # Spring constant
    L = 1  # Natural spring length
    r_init = 2
    E_init = 2*np.pi**2  # Chosen to make initial omega = 2pi
    w_init = np.sqrt(2*E_init/(m*r_init**2))
    v_t_init = w_init*r_init
    v_r_init = 0

    N_points = 256
    tmax = 2*np.pi
    ic = [r_init, v_r_init, v_t_init, w_init, E_init]

    # Solution from 0 to t=2pi
    soln = solve_ivp(spring_polar_soln, [0, 2*tmax], ic, method='DOP853',
                     dense_output=True, rtol=1e-8)

    tvals = np.linspace(0, tmax, N_points)
    soln = soln.sol(tvals)

    # Variables from solution and energy variation
    r = soln[0]
    v_r = soln[1]
    v_t = soln[2]
    w = soln[3]
    E = soln[4]
    delta_E = (E_init-E)/E

    # Finding angle theta by integrating omega
    theta = cumtrapz(w, tvals, initial=0)

    # Translating values to fit within -pi<theta<pi
    max_half_rotations = np.max(theta)/np.pi
    for i in range(int(np.ceil(max_half_rotations))+1):
        theta = np.where(theta >= np.pi, theta-2*np.pi, theta)

    # Polar projection of trajectory
    fig_pol, ax_pol = plt.subplots(subplot_kw={'projection': 'polar'})
    ax_pol.plot(theta, r)

    # Super-imposed normalised plot as functions of r_max, theta_max
    fig_super, ax_super = plt.subplots(figsize=(8, 6))
    ax_super.plot(tvals, r/np.max(r), "k", label="$r(t)$")
    ax_super.plot(tvals, w/np.max(w), "r", label="$\\omega(t)$")
    ax_super.set_ylabel("$r/r_{max}$, $\\omega/\\omega_{max}$", fontsize=16)
    ax_super.set_title("$r/r_{max}$ and $\\omega/\\omega_{max}$ over time",
                       fontsize=20)
    ax_super.set_ylim(0, 1.1)
    ax_super.legend(fontsize=14)

    # Energy-time plot
    fig_energy, ax_energy = plt.subplots(figsize=(8, 6))
    ax_energy.plot(tvals, delta_E)
    ax_energy.set_xlabel("$t$", fontsize=16)
    ax_energy.set_ylabel("$E$", fontsize=16)
    ax_energy.set_title("$(E_{init}-E)/E_{init}$ over time", fontsize=20)

    # Time dependence of radius and angle
    fig_pol, ax_pol = plt.subplots(1, 2, figsize=(10, 5))
    ax_pol[0].plot(tvals, r, "k")
    ax_pol[0].set_ylim(0, 1.1*np.max(r))
    ax_pol[0].set_xlabel("$t$", fontsize=16)
    ax_pol[0].set_ylabel("$r$", fontsize=16)
    ax_pol[0].set_title("$r(t)$", fontsize=20)
    ax_pol[1].plot(tvals, theta, "r")
    ax_pol[1].set_xlabel("$t$", fontsize=16)
    ax_pol[1].set_ylabel("$\\theta$", fontsize=16)
    ax_pol[1].set_title("$\\theta(t)$", fontsize=20)

#%% Animating the dynamics
from matplotlib.animation import FuncAnimation

if __name__ == "__main__":
    fig_ani, ax_ani = plt.subplots(subplot_kw={'projection': 'polar'})
    x = []
    y = []

    def animate(i):
        # xval = r[i]*np.cos(theta[i])
        # yval = r[i]*np.sin(theta[i])
        # x.append(xval)
        # y.append(yval)
        x.append(theta[i])
        y.append(r[i])
        ax_ani.clear()
        ax_ani.plot(x, y)
        ax_ani.set_ylim(0, 2)

    ani = FuncAnimation(fig_ani, animate, frames=N_points, interval=100,
                        repeat=False)
    plt.show()

#%% Error tolerance study
'''
Here, the impact of each solving method is demonstrated, along with the impact
of the error. We can see that higher order corrections (DOP853) provide a
higher resolution at lower errors. Low order solvers (RK23) still provide
innaccurate results.

Changing the number of points (N_points) changed the solution. This will be
due to the close encounter around the origin for the given parameter (r_init=2)
'''

if __name__ == "__main__":
    # Parameters and initial conditions
    m = 1  # Mass
    k = 1  # Spring constant
    L = 1  # Natural spring length
    r_init = 2
    E_init = 2*np.pi**2  # Chosen to make initial omega = 2pi
    w_init = np.sqrt(2*E_init/(m*r_init**2))
    v_t_init = w_init*r_init
    v_r_init = 0

    N_points = 256
    tmax = 2*np.pi
    ic = [r_init, v_r_init, v_t_init, w_init, E_init]

    error_list = np.logspace(-10, -6, 3)
    fig_pol, ax_pol = plt.subplots(3, len(error_list), figsize=(12, 8),
                                   subplot_kw={'projection': 'polar'})

    for i, method in enumerate(["RK23", "RK45", "DOP853"]):
        ax_pol[i][0].set_ylabel(method, rotation=0, labelpad=70, fontsize=20)

        for j, error in enumerate(error_list):
            # Solution from 0 to t=2pi
            soln = solve_ivp(spring_polar_soln, [0, 2*tmax], ic, method=method,
                             dense_output=True, rtol=error)

            tvals = np.linspace(0, tmax, N_points)
            soln = soln.sol(tvals)

            # Variables from solution and energy variation
            r = soln[0]
            w = soln[3]

            # Finding angle theta by integrating omega
            theta = cumtrapz(w, tvals, initial=0)

            ax_pol[i][j].plot(theta, r)
            ax_pol[0][j].set_title(error, fontsize=20)
    fig_pol.tight_layout()
