from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
wine=load_wine(as_frame=True); df=wine.frame.copy(); df["target_name"]=df["target"].map(dict(enumerate(wine.target_names)))
df.boxplot(column="alcohol",by="target_name"); plt.suptitle(""); plt.title("Alcohol by Class"); plt.show()
df.groupby("target_name")[["alcohol","malic_acid","color_intensity","proline"]].mean().plot(kind="bar"); plt.title("Selected Feature Means"); plt.show()
plt.scatter(df["flavanoids"],df["total_phenols"],c=df["target"]); plt.title("Flavanoids vs Total Phenols"); plt.show()
