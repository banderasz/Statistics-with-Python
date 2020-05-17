import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, skewnorm, lognorm


df = pd.read_csv("data/Meteorite_Landings.csv")

earth_total_area = 510_000_000
land_area = 150_000_000
not_populated_rate = 0.5
df = df.dropna(subset=["mass", "year"])

df = df.loc[(df["year"] > 1975) & (df["year"] < 2010) & (df["mass"] != 0) ]

observed_rate = land_area/earth_total_area * not_populated_rate
logmass = np.log(df["mass"])


plt.plot(df["year"], logmass, "bo")
plt.show()
plt.hist(df["year"], bins=50)
plt.show()

pd.plotting.scatter_matrix(df[["mass", "year", "reclat", "reclong"]], figsize=(7,7))
plt.show()


ms = np.linspace(-5, 20, 100)
p_skewnorm = skewnorm.fit(logmass)
pdf_skewnorm = skewnorm.pdf(ms, *p_skewnorm)
plt.hist(logmass, bins=50, alpha=0.2, density=True)
plt.plot(ms, pdf_skewnorm, c="r")
plt.show()

mass_of_doom = np.log((4/3) * np.pi * 500**3 * 1600 * 1000)  # Just using a spherical approximation and some avg density

meteor_is_doom = 1-skewnorm.cdf(mass_of_doom, *p_skewnorm)
num_events = 1000 * df["year"].value_counts().mean() / observed_rate
print(meteor_is_doom * num_events)
