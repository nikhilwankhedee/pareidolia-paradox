"""CPU-friendly image morphology descriptors."""
import numpy as np
def features(img):
 x=np.asarray(img,float); q=np.quantile(x,[.01,.1,.25,.5,.75,.9,.99]); gx=np.diff(x,axis=1); gy=np.diff(x,axis=0)
 return np.asarray(list(q)+[x.mean(),x.std(),np.mean(np.abs(gx)),np.mean(np.abs(gy)),np.mean(gx*gx),np.mean(gy*gy)],np.float32)
def matrix(images): return np.vstack([features(x) for x in images])
