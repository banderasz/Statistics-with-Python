import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

d1 = np.loadtxt("data/example_1.txt")
d2 = np.loadtxt("data/example_2.txt")
print(d1.shape, d2.shape)

plt.hist(d1, label="D1")
plt.hist(d2, label="D2")
plt.legend()
plt.ylabel("Counts")
plt.show()

bins = np.linspace(min(d1.min(), d2.min()), max(d1.max(), d2.max()), 50)
counts1, _, _ = plt.hist(d1, bins=bins, label="D1", density=True)
plt.hist(d2, bins=bins, label="D2", density=True)
plt.legend()
plt.ylabel("Counts")
plt.show()

bins = np.linspace(min(d1.min(), d2.min()), max(d1.max(), d2.max()), 50)
plt.hist([d1, d2], bins=bins, label="Stacked", density=True, histtype="barstacked", alpha=0.5)
plt.hist(d1, bins=bins, label="D1", density=True, histtype="step", lw=1)
plt.hist(d2, bins=bins, label="D2", density=True, histtype="step", ls=":")
plt.legend()
plt.ylabel("Probability")
plt.show()

dataset = pd.DataFrame({
    "value": np.concatenate((d1, d2)),
    "type": np.concatenate((np.ones(d1.shape), np.zeros(d2.shape)))
})
dataset.info()
sb.swarmplot(dataset["value"])
plt.show()


sb.swarmplot(x="type", y="value", data=dataset, size=2)
plt.show()


sb.boxplot(x="type", y="value", data=dataset, whis=3.0);
sb.swarmplot(x="type", y="value", data=dataset, size=2, color="k", alpha=0.3)
plt.show()

sb.violinplot(x="type", y="value", data=dataset);
sb.swarmplot(x="type", y="value", data=dataset, size=2, color="k", alpha=0.3);
plt.show()

sb.violinplot(x="type", y="value", data=dataset, inner="quartile", bw=0.2);
plt.show()

sd1 = np.sort(d1)
sd2 = np.sort(d2)
cdf = np.linspace(1/d1.size, 1, d1.size)

plt.plot(sd1, cdf, label="D1 CDF")
plt.plot(sd2, cdf, label="D2 CDF")
plt.hist(d1, histtype="step", density=True, alpha=0.3)
plt.hist(d2, histtype="step", density=True, alpha=0.3)
plt.legend()
plt.show()
