import pandas as pd

df = pd.read_csv("data/Diabetes.csv")
print(df.info())
print(df.describe())
df2 = df[["Glucose", "BMI", "Age"]]  # Azok az oszlopok, amikben van 0, pedig nem szabadna
mask = ~(df2 == 0).any(axis=1)  # binary mask, false, if there is a 0 in a row
df3 = df.loc[mask]  # drop the values where was a 0 in a wrong column

print(df3.groupby("Outcome").agg(["mean", "median"]))  # Show the mean en median gouped by the outcome

positive = df3.loc[df3["Outcome"] == 1]
negative = df3.loc[df3["Outcome"] == 0]
print(positive.shape, negative.shape)
