import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def func(t, y):
    print(np.shape(y))
    adot = np.exp(t)
    bdot = np.exp(2*t)
    cdot = np.exp(3*t)
    ddot = np.exp(4*t)
    ydot = np.array([[adot, bdot], [cdot, ddot]])

    return ydot

t = np.linspace(0, 2, 1024)

ic = np.array([1,1, 1,1])

# "vectorized=False" allows an (n × k) matrix instead of just (n) sized vector
soln = solve_ivp(func, [t[0], t[-1]], ic, vectorized=True,
                 dense_output=True)
sol = soln.sol(t)

plt.plot(t, sol[0], "r")
plt.plot(t, sol[1], "b")
plt.plot(t, sol[2], "g")
plt.plot(t, sol[3], "k")