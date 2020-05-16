import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

data = np.loadtxt("data/loaded_500.txt")

plt.hist(data, bins=20)
plt.show()
num_of_six = sum(data == 6)

exp_num = len(data)
ks = np.arange(40,140)
pmf_binom = st.binom.pmf(ks, exp_num, 1 / 6)

ks_ = ks[num_of_six-min(ks):]
sixes = pmf_binom[num_of_six-min(ks):]
print("Az esélye hogy ez, vagy ennél magasabb dobás bekövetkezett : {0}".format(1-sum(pmf_binom[num_of_six-min(ks):])))
plt.bar(ks, pmf_binom, label="Binomial Example (dice)", alpha=0.8)
plt.bar(ks_, sixes, label="This scenario", alpha=0.8)
plt.legend()
plt.show()
