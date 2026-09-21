"""Exact-hash grouped OOF utilities."""
import numpy as np
from sklearn.model_selection import StratifiedKFold

def assign_folds(df,n_splits=5,seed=42,hash_col='hash'):
 d=df.copy(); d['fold']=-1; groups=d.groupby(hash_col,dropna=False).indices if hash_col in d else {}
 # assign groups in label-balanced order; singleton rows are groups too
 items=[]
 for h,idx in groups.items(): items.append((h,np.asarray(idx),float(d.iloc[idx]['label'].mean())))
 rng=np.random.default_rng(seed); rng.shuffle(items); items.sort(key=lambda z: z[2])
 loads=np.zeros(n_splits); cls=np.zeros((n_splits,2))
 for _,idx,mean in items:
  score=loads + 0.5*np.abs(cls[:,1]/np.maximum(1,loads)-mean)
  f=int(np.argmin(score)); d.iloc[idx,d.columns.get_loc('fold')]=f; loads[f]+=len(idx); cls[f,1]+=mean*len(idx); cls[f,0]+=(1-mean)*len(idx)
 d['fold']=d['fold'].astype(int); return d

def assert_no_overlap(df,hash_col='hash'):
 if hash_col not in df: return True
 cross=df.groupby(hash_col)['fold'].nunique(); bad=cross[cross>1]
 if len(bad): raise ValueError(f'{len(bad)} exact hashes cross folds')
 return True

def splits(df,n_splits=5):
 for f in range(n_splits): yield np.flatnonzero(df.fold.to_numpy()!=f),np.flatnonzero(df.fold.to_numpy()==f)

def oof_fit_predict(df,X,estimator_factory):
 y=df.label.to_numpy(int); p=np.zeros(len(df))
 for tr,va in splits(df):
  m=estimator_factory(); m.fit(X[tr],y[tr]); p[va]=m.predict_proba(X[va])[:,1]
 return p
