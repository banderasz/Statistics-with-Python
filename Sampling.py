from scipy.stats import norm, uniform
from scipy.integrate import simps
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import splev, splrep

"""
The empirical distribution
"""


def pdf(x):
    return np.sin(x ** 2) + 1


xs = np.linspace(0, 4, 200)
ps = pdf(xs)
plt.plot(xs, ps)
plt.fill_between(xs, 0, ps, alpha=0.1)
plt.xlim(0, 4)
plt.ylim(0, 2)
plt.show()

"""
Rejection Sampling
Uniformly pick an x and y. Accept
if y < p(x), else pick another x and y.
Repeat.
"""
n = 100
random_x = uniform.rvs(loc=0, scale=4, size=n)
random_y = uniform.rvs(loc=0, scale=2, size=n)

plt.scatter(random_x, random_y)
plt.plot(xs, ps, c="k")
plt.fill_between(xs, 0, ps, color="w", alpha=0.1)
plt.xlim(0, 4), plt.ylim(0, 2)
plt.show()

passed = random_y <= pdf(random_x)
plt.scatter(random_x[passed], random_y[passed])
plt.scatter(random_x[~passed], random_y[~passed], marker="x", s=30, alpha=0.5)
plt.plot(xs, ps, c="w")
plt.fill_between(xs, 0, ps, color="k", alpha=0.1)
plt.xlim(0, 4), plt.ylim(0, 2)
plt.show()

"""
Test the Rejection Sampling
"""
n2 = 100000
x_test = uniform.rvs(scale=4, size=n2)
x_final = x_test[uniform.rvs(scale=2, size=n2) <= pdf(x_test)]
print(len(x_final))
from scipy.integrate import simps

plt.hist(x_final, density=True, histtype="step", label="Sampled dist")
plt.plot(xs, ps / simps(ps, x=xs), c="k", label="Empirical PDF")
plt.legend(loc=2)
plt.show()

"""
Inversion Sampling
Integrate to get the CDF from the PDF,
and then invert it (swap all xs and ys).
Uniformly sample from the CDF and
use the inverted function to get an x.
"""


def pdf(x):
    return 3 * x ** 2


def cdf(x):
    return x ** 3


def icdf(cdf):  # This is the hard part
    return cdf ** (1 / 3)


xs = np.linspace(0, 1, 100)
pdfs = pdf(xs)
cdfs = cdf(xs)
n = 2000
u_samps = uniform.rvs(size=n)
x_samps = icdf(u_samps)

fig, axes = plt.subplots(ncols=2, figsize=(10, 4))
axes[0].plot(xs, pdfs, color="k", label="PDF")
axes[0].hist(x_samps, density=True, histtype="step", label="Sampled dist", bins=50)
axes[1].plot(xs, cdfs, color="k", label="CDF")
axes[1].hlines(u_samps, 0, x_samps, linewidth=0.1, alpha=0.3)
axes[1].vlines(x_samps, 0, u_samps, linewidth=0.1, alpha=0.3)
axes[0].legend(), axes[1].legend()
axes[1].set_xlim(0, 1), axes[1].set_ylim(0, 1)
axes[0].set_xlim(0, 1), axes[0].set_ylim(0, 3)
plt.show()


def pdf(x):
    return np.sin(x ** 2) + 1


xs = np.linspace(0, 4, 10000)
pdfs = pdf(xs) / simps(pdf(xs), x=xs)
cdfs = pdfs.cumsum() / pdfs.sum()  # Dangerous

cdfs_interpol = np.insert(cdfs, 0, 0)  # insert the first element at 0
xs_interpol = np.insert(xs, 0, 0)  # insert the first element at 0

u_samps = uniform.rvs(size=10000)
x_samps = interp1d(cdfs_interpol, xs_interpol)(u_samps)  # get the interpolation

fig, axes = plt.subplots(ncols=2, figsize=(10, 4))
axes[0].hist(x_samps, density=True, histtype="step", label="Sampled dist", bins=50)
axes[0].plot(xs, pdfs, color="k", label="Analytic PDF")
axes[0].legend(loc=3), axes[0].set_xlim(0, 4)
axes[1].plot(xs_interpol, cdfs_interpol, color="k", label="Numeric CDF")
axes[1].hlines(u_samps, 0, x_samps, linewidth=0.1, alpha=0.03)
axes[1].vlines(x_samps, 0, u_samps, linewidth=0.1, alpha=0.03)
axes[1].legend(loc=2), axes[1].set_xlim(0, 4), axes[1].set_ylim(0, 1)
plt.show()
