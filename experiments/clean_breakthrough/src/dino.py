"""DINOv2 optional extractor; never silently substitutes labels or test data."""
def load_dino(name='dinov2_vits14',device=None):
 import torch
 device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
 try: model=torch.hub.load('facebookresearch/dinov2',name); return model.eval().to(device)
 except Exception as e: raise RuntimeError('DINOv2 unavailable. Enable internet or pre-cache torch.hub model.') from e
