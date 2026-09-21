import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,pandas as pd
from src.metrics import summarize
p=argparse.ArgumentParser(); p.add_argument('predictions'); p.add_argument('--metadata',required=True); p.add_argument('--output',default='artifacts/metrics.json'); a=p.parse_args(); y=pd.read_csv(a.metadata).label; d=pd.read_csv(a.predictions); r=summarize(y.to_numpy(),d.prediction.to_numpy()); Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(r,indent=2)); print(r)
