"""CLIP extractor supporting open_clip then transformers."""
def load_clip(model_name='ViT-B-32',pretrained='laion2b_s34b_b79k',device=None):
 import torch
 device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
 try:
  import open_clip; m,_,pre=open_clip.create_model_and_transforms(model_name,pretrained=pretrained); return m.eval().to(device),pre
 except Exception as e1:
  try:
   from transformers import CLIPModel,CLIPProcessor
   m=CLIPModel.from_pretrained('openai/clip-vit-base-patch32').eval().to(device); return m,CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
  except Exception as e2: raise RuntimeError('Install open_clip_torch or transformers for CLIP') from e2
