from sklearn.datasets import load_wine
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
wine=load_wine(as_frame=True); df=wine.frame
print("ANOVA p-value:",f_oneway(*[df.loc[df.target==i,"alcohol"] for i in range(3)]).pvalue)
X,y=df[wine.feature_names],df["target"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
model=Pipeline([("scaler",StandardScaler()),("classifier",LogisticRegression(max_iter=2000))])
model.fit(X_train,y_train); print("Test accuracy:",accuracy_score(y_test,model.predict(X_test)))
