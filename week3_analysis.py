from sklearn.datasets import load_wine
from scipy.stats import f_oneway,ttest_ind
wine=load_wine(as_frame=True); df=wine.frame
groups=[df.loc[df.target==i,"alcohol"] for i in range(3)]
print("ANOVA:",f_oneway(*groups))
print("Welch t-test class 0 vs 1:",ttest_ind(groups[0],groups[1],equal_var=False))
