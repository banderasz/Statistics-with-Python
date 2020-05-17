import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from chainconsumer import ChainConsumer

df_original = pd.read_csv("data/Diabetes.csv")
print(df_original.head())

cols = [c for c in df_original.columns if c not in ["Pregnancies", "Outcome"]]
df = df_original.copy()
df[cols] = df[cols].replace({0: np.NaN})
df.head()

print(df.info())
print(df.describe())

"""
Correlation
"""

pd.plotting.scatter_matrix(df, figsize=(7, 7))
plt.show()

df2 = df.dropna()
colors = df2["Outcome"].map(lambda x: "#44d9ff" if x else "#f95b4a")
pd.plotting.scatter_matrix(df2, figsize=(7,7), color=colors)
plt.show()

print(df.corr())

sb.heatmap(df.corr())
plt.show()

sb.heatmap(df.corr(), annot=True, cmap="viridis", fmt="0.2f")
plt.show()


df2 = pd.read_csv("data/height_weight.csv")
print(df2.info())
print(df2.describe())

plt.hist2d(df2["height"], df2["weight"], bins=20, cmap="magma")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.show()

hist, x_edge, y_edge = np.histogram2d(df2["height"], df2["weight"], bins=20)
x_center = 0.5 * (x_edge[1:] + x_edge[:-1])
y_center = 0.5 * (y_edge[1:] + y_edge[:-1])

plt.contour(x_center, y_center, hist)
plt.xlabel("Height")
plt.ylabel("Weight")
plt.show()

sb.kdeplot(df2["height"], df2["weight"], cmap="viridis")
plt.hist2d(df2["height"], df2["weight"], bins=20, cmap="magma", alpha=0.3)
plt.show()

sb.kdeplot(df2["height"], df2["weight"], cmap="magma", shade=True)
plt.show()

m = df2["sex"] == 1
plt.scatter(df2.loc[m, "height"], df2.loc[m, "weight"], c="#16c6f7", s=1, label="Male")
plt.scatter(df2.loc[~m, "height"], df2.loc[~m, "weight"], c="#ff8b87", s=1, label="Female")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.legend(loc=2)
plt.show()


params = ["height", "weight"]
male = df2.loc[m, params].values
female = df2.loc[~m, params].values
plt.show()


c = ChainConsumer()
c.add_chain(male, parameters=params, name="Male", kde=1, color="b")
c.add_chain(female, parameters=params, name="Female", kde=1, color="r")
c.configure(contour_labels="confidence", usetex=False)
c.plotter.plot(figsize=2.0);
plt.show()

