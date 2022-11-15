import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def bouncing_ball(t, y, coeff_rest, g):
    if y[0] < 0:
        y[1] = -coeff_rest*y[1]
    else:
        pass
    return [y[1], g]

g = -9.81
coeff_rest = 0.8

# In order [y(0), ydot(0)]
initial_conditions = [0, 50]
tf = 100
tvals = np.linspace(0, tf, 1024)

sol = solve_ivp(bouncing_ball, [0, tf], initial_conditions, t_eval=tvals, 
                args=(coeff_rest, g))

plt.plot(sol.t, sol.y[0])

#%%
# Calculates the coefficient of restitution using the height peaks
from scipy.signal import find_peaks

peak_positions = find_peaks(sol.y[0])[0]
peak_values = sol.y[0][peak_positions]
plt.plot(sol.t[peak_positions], peak_values, "rx")

coeff_experimental = peak_values[1:]/peak_values[:-1]
k = np.mean(coeff_experimental)
k_err = np.std(coeff_experimental)/np.sqrt(len(coeff_experimental))

print("Coefficient of restitution was determined to be {:.3f} ± {:.3f}".format(k, k_err))