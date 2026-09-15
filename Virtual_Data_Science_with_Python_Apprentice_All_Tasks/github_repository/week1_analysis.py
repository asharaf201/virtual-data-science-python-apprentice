from sklearn.datasets import load_wine
import pandas as pd
import matplotlib.pyplot as plt
wine=load_wine(as_frame=True); df=wine.frame.copy()
df["target_name"]=df["target"].map(dict(enumerate(wine.target_names)))
print(df.head()); print(df.shape); print(df.isnull().sum()); print("Duplicates:",df.duplicated().sum()); print(df.describe())
df["target_name"].value_counts().plot(kind="bar"); plt.title("Wine Class Distribution"); plt.show()
