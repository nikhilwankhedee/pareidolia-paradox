import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
p=argparse.ArgumentParser(); p.add_argument('--output',default='artifacts/pretrained.npz'); p.add_argument('--limit',type=int,default=0); a=p.parse_args(); raise SystemExit('Use notebook 05 or implement batched torchvision extraction with --model; no silent fallback is permitted.')
