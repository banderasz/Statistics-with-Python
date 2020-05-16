import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, anderson_ksamp
"""
sum of 10 dices
"""
rolls_obs = np.loadtxt("data/loaded_many_100.txt")
unique, count_obs = np.unique(rolls_obs, return_counts=True)
count_obs = count_obs / count_obs.sum()  # Noramlise

rolls_fair = np.random.randint(low=1, high=7, size=(500000, 10)).sum(axis=1)
unique2, count_fair = np.unique(rolls_fair, return_counts=True)
count_fair = count_fair / count_fair.sum()  # Noramlise

plt.plot(unique, count_obs, label="Data")
plt.plot(unique2, count_fair, label="Fair")
plt.legend()
plt.show()

cdf_obs = count_obs.cumsum()
cdf_fair = count_fair.cumsum()
plt.plot(unique, cdf_obs, label="Data")
plt.plot(unique2, cdf_fair, label="Fair")
plt.legend(loc=2)
plt.show()

"""
Compute the Kolmogorov-Smirnov statistic on 2 samples.

    This is a two-sided test for the null hypothesis that 2 independent samples
    are drawn from the same continuous distribution.
"""
statistic, pvalue = ks_2samp(rolls_obs, rolls_fair)
print(f"KS test statistic of {statistic:.3f}, p-value of {pvalue:.3f}")

"""
The k-sample Anderson-Darling test is a modification of the
    one-sample Anderson-Darling test. It tests the null hypothesis
    that k-samples are drawn from the same population without having
    to specify the distribution function of that population.
"""

statistic, critical_values, sig_level = anderson_ksamp([rolls_obs, rolls_fair])
print(f"AD test statistic of {statistic:.5f}, sig-level of {sig_level:.9f} (this is the p value)")
print(f"Test critical values are {critical_values}")
print("Sig-level analgous to p-value, and critical values are [25%, 10%, 5%, 2.5%, 1%]")