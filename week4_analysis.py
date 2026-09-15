from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score
wine=load_wine(as_frame=True); df=wine.frame
X,y=df[wine.feature_names],df["target"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
model=Pipeline([("scaler",StandardScaler()),("classifier",LogisticRegression(max_iter=2000))])
model.fit(X_train,y_train); pred=model.predict(X_test); prob=model.predict_proba(X_test)
print("Accuracy:",accuracy_score(y_test,pred)); print(classification_report(y_test,pred)); print(confusion_matrix(y_test,pred))
print("Weighted OVR AUC:",roc_auc_score(label_binarize(y_test,classes=[0,1,2]),prob,multi_class="ovr",average="weighted"))
