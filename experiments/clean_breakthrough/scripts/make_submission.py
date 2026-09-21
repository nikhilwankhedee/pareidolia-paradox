import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,pandas as pd
p=argparse.ArgumentParser(); p.add_argument('--test-metadata',required=True); p.add_argument('--predictions',required=True); p.add_argument('--output',default='submission.csv'); a=p.parse_args(); t=pd.read_csv(a.test_metadata); d=pd.read_csv(a.predictions); col='prediction' if 'prediction' in d else d.columns[-1]; pd.DataFrame({'image_id':t.image_id,'label':(d[col].to_numpy()>=.5).astype(int)}).to_csv(a.output,index=False); print(a.output)
