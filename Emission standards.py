import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm

data_small = np.loadtxt("data/emission_small.txt")
data_large = np.loadtxt("data/emission_large.txt")
standard = 1

plt.hist(data_small, alpha = 0.5)
plt.hist(data_large, alpha = 0.5)
plt.show()


mean, std = data_small.mean(), data_small.std()/ np.sqrt(data_small.size) #Central limit theorem
xs = np.linspace(0.5, 1.4, 200)
ps = t.pdf(xs, 9, mean, std)
prob_fail = t.sf(standard, 10, mean, std)
print(f"There is a {100 * prob_fail:.2f}% chance that the vehicle fails emission testing")

plt.hist(data_small, bins=200, label="Small")
plt.plot(xs, ps, label="Confidence of mean")
plt.axvline(1, ls=":", label="Standard")
plt.fill_between(xs, ps, 0, where=xs>=standard, alpha=0.2, color='r', label="P(fail)")
plt.legend(loc=2)
plt.show()

mean, std = data_large.mean(), data_large.std()/ np.sqrt(data_large.size) #Central limit theorem
xs = np.linspace(0.5, 1.4, 200)
ps = t.pdf(xs, 49, mean, std)
prob_fail = t.sf(standard, 9, mean, std)
print(f"There is a {100 * prob_fail:.2f}% chance that the vehicle fails emission testing")

plt.hist(data_small, bins=200, label="Small")
plt.plot(xs, ps, label="Confidence of mean")
plt.axvline(1, ls=":", label="Standard")
plt.fill_between(xs, ps, 0, where=xs>=standard, alpha=0.2, color='r', label="P(fail)")
plt.legend(loc=2)
plt.show()

mean, std = data_large.mean(), data_large.std()/ np.sqrt(data_large.size) #Central limit theorem
xs = np.linspace(0.5, 1.4, 200)
ps = norm.pdf(xs, mean, std)
prob_pass = norm.cdf(standard, mean, std)
print(f"There is a {100 * (1-prob_pass):.2f}% chance that the vehicle fails emission testing")

plt.hist(data_small, bins=200, label="Small")
plt.plot(xs, ps, label="Confidence of mean")
plt.axvline(1, ls=":", label="Standard")
plt.fill_between(xs, ps, 0, where=xs>=standard, alpha=0.2, color='r', label="P(fail)")
plt.legend(loc=2)
plt.show()