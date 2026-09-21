import argparse
p=argparse.ArgumentParser(description='CLIP extraction is GPU/network dependent; use notebook 07.'); p.add_argument('--output',default='artifacts/clip.npz'); p.parse_args(); raise SystemExit('Run notebook 07 with open_clip or transformers available, then cache embeddings.')
