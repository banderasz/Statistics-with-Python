import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from scipy.stats import chi2

data = np.loadtxt("data/loaded_500.txt")
unique, counts = np.unique(data, return_counts=True)
plt.hist(data, bins=50)
plt.show()


# generated = np.random.randint(low=1, high=7, size=data.size)
# _, expected = np.unique(generated, return_counts=True)
# plt.hist(generated, bins=50)
# plt.show()
"""
Test a discrete distribution against expected values.
"""
expected = data.size / 6
chisq, p = chisquare(counts, expected)
print(f"We have a chi2 of {chisq:.2f} with a p-value of {p:.3f}")


chi2s = np.linspace(0, 15, 500)
prob = chi2.pdf(chi2s, 5)

plt.plot(chi2s, prob, label="Distribution")
plt.axvline(chisq, label="$\chi2$", ls="--")
plt.fill_between(chi2s, prob, 0, where=(chi2s>=chisq), alpha=0.1)
plt.legend()
plt.show()
print(f"Our p-value is {chi2.sf(chisq, 5):.3f}")