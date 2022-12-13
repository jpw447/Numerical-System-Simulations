#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
if __name__ == "__main__":
    # Constants
    G = 6.67408e-11
    M_planet = 5.972e24
    M_star = 1.989e30
    AU = 1.496e11
    
    # Conversion factors such that x' = x/x_0 etc. for new variable x'
    # [x] = AU, [t] = Years, [v] = AU/yr
    x_0 = 1.496e11                  # m to AU
    t_0 = 60**2 * 24 * 365.25       # s to years
    
    # Get division by zero otherwise
    zero = 0.000001
    
    # Intitial conditions
    x_star_init = zero
    y_star_init = zero
    vx_star_init = zero
    vy_star_init = zero
    x_planet_init = 1*AU
    y_planet_init = zero
    vx_planet_init = zero
    separation = np.sqrt( (x_star_init - x_planet_init)**2 + (y_star_init - y_planet_init)**2 )
    vy_planet_init = np.sqrt(G*M_star/separation)
    
    # Time array
    year_constant = 60**2 * 24 * 365.25
    number_of_years = 1
    t_max = number_of_years * year_constant
    t_array = np.linspace(0, t_max, 1000)
    # Lists containin intial conditions (parameters) and important constants.
    # These appear in a certain order here, and the order must be adheredt to 
    # everywhere else you create a list like this - in the function passed to
    # odeint in both input and output, and what odeint outputs.
    initial_parameters =[x_star_init, y_star_init, x_planet_init, y_planet_init,
                         vx_star_init, vy_star_init, vx_planet_init, vy_planet_init]
    constants = [G, M_planet, M_star]
    def field_function(parameters, t, constants):
        '''
        Function that takes input parameters (initial conditions) and constants, as well as at time array.
        Returns a list containing the field of differential equations for each derivative.
        Args:
            parameters: list with initial conditions, containing positions and velocities of 2 bodies
            t: time array used by ode_int
            constants: list containing constants such as Gravitational Constant and masses
        Returns:
            field: list containing the derivatives for the system
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
        # Correctrdering of elements is critical for odeint to work
        field = [x_star_prime, y_star_prime, x_planet_prime, y_planet_prime,
                vx_star_prime, vy_star_prime, vx_planet_prime, vy_planet_prime]
        return field
    # sol = array containing 8 columns for x_star, y_star etc. in order they appear in field_function. Each row is t value
    # Columns:
    # 0: x_star
    # 1: y_star
    # 2: x_planet
    # 3: y_planet
    # 4: vx_star
    # 5: vy_star
    # 6: vx_planet
    # 7: vy_planet
    # Passing function to odeint and retrieving planet and star positions
    sol = odeint(field_function, initial_parameters, t_array, args=(constants,)) 
    # Converting to AU as well   
    x_planet, y_planet = sol[:, 2]/AU, sol[:, 3]/AU

    # Creating figures and axes
    fig_planet = plt.figure()
    fig_planet = plt.figure(figsize=(8,6))
    ax_planet = fig_planet.gca()
    fig_sinusoid = plt.figure()
    fig_sinusoid = plt.figure(figsize=(8,6))
    ax_sinusoid = fig_sinusoid.gca()

    # Creating plots
    ax_planet.plot(x_planet, y_planet, 'k')
    ax_planet.set_title("Planet position", fontsize=20)
    ax_planet.set_xlabel("$x_{\\bigoplus}$ (AU)", fontsize=16)
    ax_planet.set_ylabel("$y_{\\bigoplus}$ (AU)", fontsize=16)
    ax_planet.set_aspect('equal')

    t_array = t_array / year_constant # Converting time back to years

    ax_sinusoid.plot(t_array, x_planet, 'k')
    ax_sinusoid.set_title("Planet Position Versus Time", fontsize=20)
    ax_sinusoid.set_xlabel("$t$ (years)", fontsize=16)
    ax_sinusoid.set_ylabel("$x_{\\bigoplus}$ (AU)", fontsize=16)
    plt.show()