import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import splev, splrep
from scipy.integrate import simps
from statsmodels.distributions.empirical_distribution import ECDF

xs = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
      5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0])
ys = np.array([0.2, 0.165, 0.167, 0.166, 0.154, 0.134, 0.117,
      0.108, 0.092, 0.06, 0.031, 0.028, 0.048, 0.077,
      0.103, 0.119, 0.119, 0.103, 0.074, 0.038, 0.003])
ys = ys/sum(ys)

plt.scatter(xs, ys)
plt.xlabel("x")
plt.ylabel("Observed PDF")
plt.show()


"""
Interpolations
"""

x = np.linspace(min(xs), max(xs), 1000)
y1 = interp1d(xs, ys)(x)
y2 = interp1d(xs, ys, kind="nearest")(x)
y3 = interp1d(xs, ys, kind="quadratic")(x)
y4 = interp1d(xs, ys, kind="cubic")(x)
y5 = splev(x, splrep(xs, ys))

plt.scatter(xs, ys, s=30, label="Data", c="w")
plt.plot(x, y1, label="Linear (default)")
plt.plot(x, y2, label="Nearest", alpha=0.2)
plt.plot(x, y3, label="Quadratic", ls='-')
plt.plot(x, y4, label="Cubic", ls='-')
plt.plot(x, y5, label="Spline", ls='-', alpha=1)
plt.legend()
plt.show()



"""
Calculate probability in interval and cdf
"""
def get_prob(xs, ys, a, b, resolution=1000):
    """
    Probability of the pdf with xs and ys between a and b
    """
    x_norm = np.linspace(min(xs), max(xs), resolution)
    y_norm = interp1d(xs, ys, kind="quadratic")(x_norm)
    normalisation = simps(y_norm, x=x_norm)
    x_vals = np.linspace(a, b, resolution)
    y_vals = interp1d(xs, ys, kind="quadratic")(x_vals)
    return simps(y_vals, x=x_vals) / normalisation

def get_cdf(xs, ys, v):
    return get_prob(xs, ys, min(xs), v)

def get_sf(xs, ys, v):
    return 1 - get_cdf(xs, ys, v)

v1, v2 = 6, 9.3
area = get_prob(xs, ys, v1, v2)

plt.scatter(xs, ys, s=30, label="Data", color="w")
plt.plot(x, y3, linestyle="-", label="Interpolation")
plt.fill_between(x, 0, y3, where=(x>=v1)&(x<=v2), alpha=0.2)
plt.annotate(f"p = {area:.3f}", (7, 0.05))
plt.legend()
plt.show()


x_new = np.linspace(min(xs), max(xs), 100)
cdf_new = [get_cdf(xs, ys, i) for i in x_new]
cheap_cdf = y3.cumsum() / y3.sum()

plt.plot(x_new, cdf_new, label="Interpolated CDF")
plt.plot(x, cheap_cdf, label="Super cheap CDF for specific cases")
plt.legend()
plt.ylabel("CDF")
plt.xlabel("x")
plt.show()