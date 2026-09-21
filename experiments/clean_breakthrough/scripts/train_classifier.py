import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from _common import data
from src.azimuth import harmonic
from src.classifiers import make
from src.metrics import summarize
from src.validation import splits
p=argparse.ArgumentParser(); p.add_argument('--data-root'); p.add_argument('--output',default='artifacts'); p.add_argument('--hashes'); p.add_argument('--folds',type=int,default=5); p.add_argument('--seed',type=int,default=42); p.add_argument('--model',default='logistic'); a=p.parse_args()
d,tr,te=data(a); X=harmonic(tr.azimuth,3); y=tr.label.to_numpy(); o=np.zeros(len(tr))
for ti,vi in splits(tr,a.folds): o[vi]=make(a.model,a.seed).fit(X[ti],y[ti]).predict_proba(X[vi])[:,1]
out=Path(a.output); out.mkdir(parents=True,exist_ok=True); pd.DataFrame({'image_id':tr.image_id,'prediction':o,'fold':tr.fold}).to_csv(out/'oof_predictions.csv',index=False); (out/'metrics.json').write_text(json.dumps(summarize(y,o),indent=2)); print(summarize(y,o))
