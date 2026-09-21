import argparse
p=argparse.ArgumentParser(description='DINOv2 extraction is GPU/network dependent; use notebook 06.'); p.add_argument('--output',default='artifacts/dino.npz'); p.parse_args(); raise SystemExit('Run notebook 06 with internet enabled to download DINOv2, then cache embeddings.')
