import json,random,os
from pathlib import Path
import numpy as np
def seed_all(seed=42):
 random.seed(seed); np.random.seed(seed); os.environ['PYTHONHASHSEED']=str(seed)
 try:
  import torch; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
 except Exception: pass
def read_yaml(path):
 try:
  import yaml; return yaml.safe_load(Path(path).read_text())
 except ImportError:
  return {}
def dump_json(obj,path): Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(json.dumps(obj,indent=2,default=str))
def ensure_dir(path): Path(path).mkdir(parents=True,exist_ok=True); return Path(path)
