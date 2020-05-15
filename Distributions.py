import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np

xs = np.linspace(-5, 10, 2000)
ks = np.arange(50)

pmf_binom = st.binom.pmf(ks, 50, 0.25)
plt.bar(ks, pmf_binom, label="Binomial Example (dice)", alpha=0.8)

pmf_poisson = st.poisson.pmf(ks, 30)
plt.bar(ks, pmf_poisson, label="Poisson Example (car crash)", alpha=0.8)
plt.legend()
plt.show()

print(st.binom.pmf(10, 50, 0.25))
print(st.poisson.pmf(50, 30))



pdf_uniform = st.uniform.pdf(xs, -4, 10)
plt.plot(xs, pdf_uniform, label="Uniform(-4,6)")

pdf_normal = st.norm.pdf(xs, loc=5, scale=2)
plt.plot(xs, pdf_normal, label="Normal(5, 2)")

pdf_exponential = st.expon.pdf(xs, loc=-2, scale=2)
plt.plot(xs, pdf_exponential, label="Exponential(0.5)")

pdf_studentt = st.t.pdf(xs, 1)
plt.plot(xs, pdf_studentt, label="Student-t(1)")

pdf_lognorm = st.lognorm.pdf(xs, 1)
plt.plot(xs, pdf_lognorm, label="Lognorm(1)")

pdf_skewnorm = st.skewnorm.pdf(xs, -6)
plt.plot(xs, pdf_skewnorm, label="Skewnorm(5)")

plt.legend()
plt.ylabel("Prob")
plt.xlabel("x")
plt.show()