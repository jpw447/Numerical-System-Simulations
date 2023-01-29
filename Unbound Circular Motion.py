'''
Code to simulate a body under a centripetal force pushing it radially outwards.
Energy seems to be conserved (particularly as method is changed from RK23 to 
DOP853 and relative tolerance is changed to ~1e-8), so it's likely to be correct
'''


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def func(t, parameters):
    r, v_r, v_t, w, theta = parameters

    dr    = v_r
    dv_r = w**2 * r
    dv_t = w*r
    dw = np.sqrt(2*E/m)*dr*(-1/(r**2))
    dtheta = w
    
    return  [dr, dv_r, dv_t, dw, dtheta]
    

m = 1
r_init = 1
E = np.pi
w_init = np.sqrt(2*E/(m*r_init**2))
v_t_init = w_init*r_init
v_r_init = 0
theta_init = 0


ic = [r_init, v_r_init, v_t_init, w_init, theta_init]

soln = solve_ivp(func, [0, 2*np.pi], ic, method='DOP853', dense_output=True,
                 rtol = 1e-8)

tvals = np.linspace(0, 2*np.pi, 128)
soln = soln.sol(tvals)

r = soln[0]
v_r = soln[1]
v_t = soln[2]
w = soln[3]
theta = soln[4]
delta_E = (0.5*m*r**2*w**2 - E)/E

# energy not conserved
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
plt.plot(tvals, delta_E)

