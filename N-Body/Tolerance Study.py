import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

'''
Issues:
    odeint and solve_ivp provide the same, correct solution. However, solve_ivp
    provides different numerical solvers (mostly Runge-Kutta following different
    orders and corrections) which have certain in-built tolerances.
'''


def ivp_function(t, parameters, constants):
    '''
    Parameters
    ----------
    t : float
        time step value used by solve_ivp.
    parameters : np.ndarray
        List of values use for the system. Order must follow the line where
        parameters is assigned to multiple variables.
    constants : np.ndarray
        List of constants used for calculations (G, masses etc.)

    Returns
    -------
    field : np.ndarray
        THe change in all values that were passed through parameters, as
        calculated by the field of equations. Used to find values at the next
        time step using the selected numerical solver method.
    '''

    x_star, y_star, x_planet, y_planet, vx_star, vy_star, vx_planet, vy_planet = parameters
    G, M_planet, M_star = constants

    x_rel = x_star - x_planet
    y_rel = y_star - y_planet
    r_cubed = (x_rel*x_rel + y_rel*y_rel)**(1.5)

    # ODE system
    x_star_prime = vx_star
    vx_star_prime = -(G*M_planet/r_cubed) * x_rel
    y_star_prime = vy_star
    vy_star_prime = -(G*M_planet/r_cubed) * y_rel
    x_planet_prime = vx_planet
    vx_planet_prime = (G*M_star/r_cubed) * x_rel
    y_planet_prime = vy_planet
    vy_planet_prime = (G*M_star/r_cubed) * y_rel

    field = [x_star_prime, y_star_prime, x_planet_prime, y_planet_prime,
             vx_star_prime, vy_star_prime, vx_planet_prime, vy_planet_prime]
    return field


G = 6.67408e-11
M_planet = 5.972e24
M_star = 1.989e30
AU = 1.496e11
vy_p_init = np.sqrt(G*M_star/AU)

num_years = 5
year_constant = 365*24*60**2
t0 = 0
tf = num_years*year_constant

positions = [0, 0, AU, 0]
velocities = [0, 0, 0, np.sqrt(G*M_star/AU)]
ic = np.array(positions + velocities)
constants = [G, M_planet, M_star]

tvals = np.linspace(t0, tf, 2**12)

tolerances = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8]

fig, ax_tolerance = plt.subplots(figsize=(8, 6))
ax_tolerance.set_title("2 Body Solver of different tolerances", fontsize=16)
ax_tolerance.set_xlabel("$t$", fontsize=14)
ax_tolerance.set_ylabel("$\\Delta E/E_{0}$", fontsize=14)

for tolerance in tolerances:
    soln = solve_ivp(ivp_function, [t0, tf], ic, args=(constants,), method="RK45",
                     dense_output=True, rtol=tolerance)

    # solve_ivp solution
    sol = soln.sol(tvals)
    x_s = sol[0]
    y_s = sol[1]
    x_p = sol[2]
    y_p = sol[3]
    vx_s = sol[4]
    vy_s = sol[5]
    vx_p = sol[6]
    vy_p = sol[7]
    separation = ((x_s - x_p)**2 + (y_s - y_p)**2)**0.5

    KE = 0.5*M_planet*(vx_p**2 + vy_p**2) + 0.5*M_star*(vx_s**2 + vy_s**2)
    PE = -G*M_planet*M_star/separation
    E = KE + PE
    delta_E = (E-E[0])/E[0]

    ax_tolerance.plot(tvals/year_constant, delta_E, label="rtol="+str(tolerance))

ax_tolerance.legend()

#%%
'''
Measuring the time taken for each solver method to solve the system, with
specified tolerance levels.
'''