import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
from scipy.interpolate import interp1d

data = np.loadtxt("data/dataset.txt")
plt.hist(data, bins=100)
plt.show()

"""
Calculate mean, median
"""
outlier = np.insert(data, 0, 5000)
plt.hist(data, label="Data", bins=100);
plt.axvline(np.mean(data), ls="--", label="Mean Data")
plt.axvline(np.median(data), ls=":", label="Median Data")
plt.axvline(np.mean(outlier), c='r', ls="--", label="Mean Outlier", alpha=0.7)
plt.axvline(np.median(outlier), c='r', ls=":", label="Median Outlier", alpha=0.7)
plt.legend()
plt.xlim(0, 20)
plt.show()

"""
Calculate gaussian KDE (Kernel density estimation is a way to estimate the probability density
    function (PDF) of a random variable in a non-parametric way.)
    
    Also calculate mode.
"""
kde = st.gaussian_kde(data)
xvals = np.linspace(data.min(), data.max(), 1000)
yvals = kde(xvals)
mode = xvals[yvals.argmax()]
plt.hist(data, bins=1000, density=True, label="Data hist")
plt.plot(xvals, yvals, label="KDE")
plt.axvline(mode, label="Mode", c='r')
plt.legend()
plt.show()

plt.hist(data, bins=100, label="Data", alpha=0.5)
plt.axvline(data.mean(), label="Mean", ls="--", c='#f9ee4a')
plt.axvline(np.median(data), label="Median", ls="-", c='#44d9ff')
plt.axvline(mode, label="Mode", ls=":", c='#f95b4a')
plt.legend()
plt.show()

"""
Approximate with gauss dist.
"""
xs = np.linspace(data.min(), data.max(), 100)
ys = st.norm.pdf(xs, loc=np.mean(data), scale=np.std(data))

plt.hist(data, bins=50, density=True, histtype="step", label="Data")
plt.plot(xs, ys, label="Normal approximation")
plt.legend()
plt.ylabel("Probability")
plt.show()

"""
Approximate with gauss dist + skew.
"""
xs = np.linspace(data.min(), data.max(), 100)
ys1 = st.norm.pdf(xs, loc=np.mean(data), scale=np.std(data))
ys2 = st.skewnorm.pdf(xs, st.skew(data), loc=np.mean(data), scale=np.std(data))

plt.hist(data, bins=50, density=True, histtype="step", label="Data")
plt.plot(xs, ys1, label="Normal approximation")
plt.plot(xs, ys2, label="Skewnormal approximation")
plt.legend()
plt.ylabel("Probability")
plt.show()

"""
Approximate with percentiles.
"""
ps = np.linspace(0, 100, 10)
x_p = np.percentile(data, ps)

xs = np.sort(data)
ys = np.linspace(0, 1, len(data))

plt.plot(xs, ys * 100, label="ECDF")
plt.plot(x_p, ps, label="Percentiles", marker=".", ms=10)
plt.legend()
plt.ylabel("Percentile")
plt.show()

"""
Calculate the percentile of the cdf function
"""
ps = 100 * st.norm.cdf(np.linspace(-3, 3, 50))
ps = np.concatenate(([0], ps, [100]))  # There is a bug in the insert way of doing it, this is better
x_p = np.percentile(data, ps)

xs = np.sort(data)
ys = np.linspace(0, 1, len(data))

plt.plot(xs, ys * 100, label="ECDF")
plt.plot(x_p, ps, label="Percentiles", marker=".", ms=10)
plt.legend()
plt.ylabel("Percentile")
plt.show()

n = int(1e6)
u = np.random.uniform(size=n)
samp_percentile_1 = interp1d(ps / 100, x_p)(u)

_, bins, _ = plt.hist(data, bins=50, density=True, alpha=0.3, label="Data")
plt.hist(samp_percentile_1, bins=bins, density=True, histtype="step", label="Percentiles")
plt.ylabel("Probability")
plt.legend()
plt.show()


"""
Covariance in multidimension dataset
"""

dataset = pd.read_csv("data/height_weight.csv")[["height", "weight"]]
covariance = dataset.cov()
print(covariance)
