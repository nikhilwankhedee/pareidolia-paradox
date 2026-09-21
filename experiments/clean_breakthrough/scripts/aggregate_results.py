import argparse,glob,pandas as pd,json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--inputs',nargs='*',default=glob.glob('artifacts/*.json')); p.add_argument('--output',default='results/leaderboard.csv'); a=p.parse_args(); rows=[]
for f in a.inputs:
 try: x=json.load(open(f)); x['artifact']=f; rows.append(x)
 except Exception: pass
Path(a.output).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.output,index=False); print('wrote',a.output)
