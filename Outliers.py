import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal

d1 = np.loadtxt("data/outlier_1d.txt")
d2 = np.loadtxt("data/outlier_2d.txt")
d3 = np.loadtxt("data/outlier_curve.txt")
print(d1.shape, d2.shape)


"""
Visualize the data
"""
plt.scatter(d1, np.random.normal(7, 0.2, size=d1.size), s=1, alpha=0.5)
plt.scatter(d2[:, 0], d2[:, 1], s=5)
plt.show()
plt.plot(d3[:, 0], d3[:, 1])
plt.show()


"""
FInd outliers by z_score (standard deviation index) in 1D
"""
mean, std = np.mean(d1), np.std(d1)
z_score = np.abs((d1 - mean) / std)
threshold = 3
good = z_score < threshold

print(f"Rejection {(~good).sum()} points")

print(f"z-score of 3 corresponds to a prob of {100 * 2 * norm.sf(threshold):0.2f}%")
visual_scatter = np.random.normal(size=d1.size)
plt.scatter(d1[good], visual_scatter[good], s=2, label="Good", color="#4CAF50")
plt.scatter(d1[~good], visual_scatter[~good], s=8, label="Bad", color="#F44336")
plt.legend()
plt.show()


"""
FInd outliers by z_score (standard deviation index) in 1D
"""
mean, cov = np.mean(d2, axis=0), np.cov(d2.T)
logical_func = multivariate_normal(mean, cov) # model it with a 2d normal distribution
probs = multivariate_normal(mean, cov).pdf(d2)
good = probs > 0.01 / 100 # good if the probability of the value is greater then 0.01%

plt.scatter(d2[good, 0], d2[good, 1], s=2, label="Good", color="#4CAF50")
plt.scatter(d2[~good, 0], d2[~good, 1], s=8, label="Bad", color="#F44336")
plt.legend()
plt.show()


"""
Find outlieres by iterative polyfit
"""
xs, ys = d3.T
p = np.polyfit(xs, ys,deg=5)
ps = np.polyval(p, xs)
plt.plot(xs, ys, ".", label="Data", ms=1)
plt.plot(xs, ps, label="Bad poly fit")
plt.legend()
plt.show()

x, y = xs.copy(), ys.copy()
for i in range(10):
    p = np.polyfit(x, y, deg=5)
    ps = np.polyval(p, x)
    good = y - ps < 2

    x_bad, y_bad = x[~good], y[~good]
    x, y = x[good], y[good]
    plt.plot(x, y, ".", label="Used Data", ms=1)
    plt.plot(x, np.polyval(p, x), label=f"Poly fit {i}")
    plt.plot(x_bad, y_bad, ".", label="Not used Data", ms=5, c="r")
    plt.legend()
    plt.show()

    if (~good).sum() == 0:
        break
