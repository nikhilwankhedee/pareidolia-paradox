import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

def make(name='logistic',seed=42):
 if name in ('hgb','histgradientboosting'): return HistGradientBoostingClassifier(max_iter=200,random_state=seed)
 return make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=seed,C=1.0))
def fit_predict(Xtr,ytr,Xva,name='logistic',seed=42):
 m=make(name,seed); m.fit(Xtr,ytr); return m.predict_proba(Xva)[:,1],m
