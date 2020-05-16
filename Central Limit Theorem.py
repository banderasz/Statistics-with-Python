import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, skewnorm, uniform
from scipy.stats import norm, uniform
from scipy.integrate import simps
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import splev, splrep

def pdf(x):
    return 3 * x ** 5

xs = np.linspace(0, 1, 100)
pdfs = pdf(xs) / simps(pdf(xs), x=xs)
cdfs = pdfs.cumsum() / pdfs.sum()  # Dangerous

cdfs_interpol = np.insert(cdfs, 0, 0)  # insert the first element at 0
xs_interpol = np.insert(xs, 0, 0)  # insert the first element at 0

u_samps = uniform.rvs(size=100)
x_samps = interp1d(cdfs_interpol, xs_interpol)(u_samps)

plt.plot(xs, pdfs, color="k", label="PDF")
plt.hist(x_samps, density=True, histtype="step", label="Sampled dist", bins=50)
plt.show()


def get_data(n):
    u_samps = uniform.rvs(size=n)
    data = interp1d(cdfs_interpol, xs_interpol)(u_samps)
    np.random.shuffle(data)
    return data

means = [get_data(100).mean() for i in range(1000)]
plt.hist(means, bins=50)
print(np.std(means))
plt.show()
