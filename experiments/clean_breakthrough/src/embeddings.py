"""Embedding extraction primitives with cache-safe deterministic keys."""
from pathlib import Path
import hashlib, json, numpy as np

def cache_key(paths,model,config=None):
 s=json.dumps({'model':model,'config':config or {},'files':[str(x) for x in paths]},sort_keys=True); return hashlib.sha256(s.encode()).hexdigest()[:20]
def save(path,X,ids=None):
 Path(path).parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(path,features=np.asarray(X,dtype=np.float32),image_id=np.asarray(ids if ids is not None else [],dtype=str))
def load(path):
 z=np.load(path,allow_pickle=False); return z['features'],z['image_id'] if 'image_id' in z else None
