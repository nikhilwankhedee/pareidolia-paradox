"""Torchvision modern encoder loader."""
def load_encoder(name='resnet18',weights='DEFAULT',device=None):
 import torch, torchvision
 fn=getattr(torchvision.models,name)
 try: w=getattr(torchvision.models, f'{name.upper()}_Weights').DEFAULT if weights else None
 except Exception: w=None
 m=fn(weights=w); return m.eval().to(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
