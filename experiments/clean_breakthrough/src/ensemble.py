import numpy as np
def weighted_average(predictions,weights=None):
 P=np.vstack([np.asarray(x) for x in predictions]); w=np.ones(P.shape[0])/P.shape[0] if weights is None else np.asarray(weights)/np.sum(weights); return np.average(P,axis=0,weights=w)
def rank_average(predictions):
 P=np.vstack(predictions); return np.mean(np.argsort(np.argsort(P,axis=1),axis=1)/(P.shape[1]-1),axis=0)
