import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint

'''
Issues:
    odeint and solve_ivp provide the same, correct solution. Solve_ivp doesn't
    provide the correct solution for the RK45 method. However, odeint is marginally
    faster:
        odeint - 5.55 ms ± 23.6 µs
        RK23 - 13.3 ms ± 296 µs
'''


def ivp_function(t, parameters, constants):
    '''
    Function passed to solve_ivp
    '''
    x_star, y_star, x_planet, y_planet, vx_star, vy_star, vx_planet, vy_planet = parameters
    G, M_planet, M_star = constants
    # Relative positions
    x_rel = x_star - x_planet
    y_rel = y_star - y_planet
    r_cubed = (x_rel*x_rel + y_rel*y_rel)**(1.5)
    x_star_prime = vx_star
    vx_star_prime = -(G*M_planet/r_cubed) * x_rel
    y_star_prime = vy_star
    vy_star_prime = -(G*M_planet/r_cubed) * y_rel
    x_planet_prime = vx_planet
    vx_planet_prime = (G*M_star/r_cubed) * x_rel
    y_planet_prime = vy_planet
    vy_planet_prime = (G*M_star/r_cubed) * y_rel
    
    # List containing the right hand side of the coupled ODE system
    # Correct ordering of elements is critical for odeint to work
    field = [x_star_prime, y_star_prime, x_planet_prime, y_planet_prime,
            vx_star_prime, vy_star_prime, vx_planet_prime, vy_planet_prime]
    return field


def field_function(parameters, t, constants):
    '''
    Function passed to odeint
    '''    
    x_star, y_star, x_planet, y_planet, vx_star, vy_star, vx_planet, vy_planet = parameters
    G, M_planet, M_star = constants
    # Relative positions
    x_rel = x_star - x_planet
    y_rel = y_star - y_planet
    r_cubed = (x_rel*x_rel + y_rel*y_rel)**(1.5)
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
t0 = 0
tf = num_years*365*24*60*60

positions = [0, 0, AU, 0]
velocities = [0, 0, 0, np.sqrt(G*M_star/AU)]
ic = np.array(positions + velocities)
constants = [G, M_planet, M_star]

tvals = np.linspace(t0, tf, 2**12)

''' 
Methods:
    DOP853
    RK23
    RK45
    Radau
    BDF
    LSODA
from https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
'''
solver_method = "DOP853"

soln = solve_ivp(ivp_function, [t0, tf], ic, args=(constants,), method=solver_method,
                 dense_output=True)
soln_ode = odeint(field_function, ic, tvals, args=(constants,))

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

# odeint solution
x_star, y_star = soln_ode[:, 0], soln_ode[:, 1]
x_planet, y_planet = soln_ode[:, 2], soln_ode[:, 3]
vx_star, vy_star = soln_ode[:, 4], soln_ode[:, 5]
vx_planet, vy_planet = soln_ode[:, 6], soln_ode[:, 7]

separation = ((x_s - x_p)**2 + (y_s - y_p)**2)**0.5
KE_ode = 0.5*M_planet*(vx_planet**2 + vy_planet**2) + 0.5*M_star*(vx_star**2 + vy_star**2)
PE_ode = -G*M_planet*M_star/separation
E_ode = KE_ode + PE_ode
delta_E_ode = (E_ode-E_ode[0])/E_ode[0]

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("2 Body Solver ("+str(solver_method)+" solve_ivp method)", fontsize=16)
ax[0].plot(tvals, delta_E, "r", label="solve_ivp")
ax[0].plot(tvals, delta_E_ode, "b", label="odeint")
ax[0].set_title("Percentage Energy Change with Time", fontsize=12)
ax[0].set_xlabel("$t$", fontsize=16)
ax[0].set_ylabel("$\\Delta E/E_{0}$", fontsize=16)
ax[0].legend(fontsize=12, loc="upper left")

ax[1].plot(tvals, x_p, "r", label="solve_ivp")
ax[1].plot(tvals, x_planet, "b", label="odeint")
ax[1].set_title("$x$ Position with Time", fontsize=12)
ax[1].set_xlabel("$t$", fontsize=16)
ax[1].set_ylabel("$x$", fontsize=16)
ax[1].legend(fontsize=12, loc="upper left")

