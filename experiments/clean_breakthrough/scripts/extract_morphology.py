import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.dataset import discover_dataset,load_image
from src.morphology import features
p=argparse.ArgumentParser(); p.add_argument('--data-root'); p.add_argument('--output',default='artifacts/morphology.npz'); p.add_argument('--limit',type=int,default=0); a=p.parse_args(); d=discover_dataset(a.data_root); ids=d['train'].image_id.tolist(); ids=ids[:a.limit] if a.limit else ids; X=np.vstack([features(load_image(Path(d['train_images'])/i)) for i in ids]); Path(a.output).parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(a.output,features=X,image_id=np.asarray(ids)); print(X.shape)
