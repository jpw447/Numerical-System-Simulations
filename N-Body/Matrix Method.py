import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint

'''
In the first cell, RK order 2 correction order 3 is used to calculate the
orbit of the earth around the sun for 5 years, with default tolerance levels.

In the second cell, different numerical solvers are investigated for their 
effects on the solution, with a specified tolerance level.
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


# def matrix_method(parameters, t, constants):
#     '''
#     The classic 2-body problem written using matrics to, hopefully, cut down
#     on calculation time
#     '''    
#     x_star, y_star, x_planet, y_planet, vx_star, vy_star, vx_planet, vy_planet = parameters
#     G, M_planet, M_star = constants
    
#     # Position vectors
#     P = np.array([x_planet, y_planet])
#     S = np.array([x_star, y_star])
#     Vp = np.array([vx_planet, vy_planet])
#     Vs = np.array(vx_star, vy_star])
    
#     # Vectorising vectors
#     positions = np.array([P, S])
#     velocities = np.array([Vp, Vs])
    
    
    

#     field = 
#     return field


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

solver_method = "RK23"

soln = solve_ivp(ivp_function, [t0, tf], ic, args=(constants,), method=solver_method,
                 dense_output=True)

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

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("2 Body Solver ("+str(solver_method)+" solve_ivp method)", fontsize=16)
ax[0].plot(tvals/year_constant, delta_E, "r", label="solve_ivp")
ax[0].set_title("Percentage Energy Change with Time", fontsize=12)
ax[0].set_xlabel("$t$ (years)", fontsize=16)
ax[0].set_ylabel("$\\Delta E/E_{0}$", fontsize=16)
ax[0].legend(fontsize=12, loc="upper left")

ax[1].plot(tvals/year_constant, x_p/AU, "r", label="solve_ivp")
ax[1].plot(tvals/year_constant, np.cos(2*np.pi/(tf/num_years)*tvals), "b", label="$\\sin(\\omega t)$")
ax[1].set_title("$x$ Position with Time", fontsize=12)
ax[1].set_xlabel("$t$ (years)", fontsize=16)
ax[1].set_ylabel("$x$ (AU)", fontsize=16)
ax[1].legend(fontsize=12, loc="upper left")

