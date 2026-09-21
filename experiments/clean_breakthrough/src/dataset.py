"""Portable Kaggle/local dataset discovery and leakage-safe image access."""
from pathlib import Path
import os, hashlib
import numpy as np
import pandas as pd

def discover_dataset(root=None):
    candidates=[]
    if root: candidates.append(Path(root))
    env=os.getenv('PAREIDOLIA_DATA_ROOT')
    if env: candidates.append(Path(env))
    candidates += [Path('/kaggle/input'), Path.cwd(), Path(__file__).resolve().parents[3]]
    roots=[]
    for base in candidates:
        if not base.exists(): continue
        for p in ([base] if (base/'Train'/'train_metadata.csv').exists() or (base/'train_metadata.csv').exists() else base.rglob('train_metadata.csv')):
            roots.append(p.parent if p.name=='train_metadata.csv' else p)
    for r in roots:
        train_meta=r/'train_metadata.csv'; test_meta=r/'test_metadata.csv'
        if not train_meta.exists() or not test_meta.exists():
            if (r/'Train'/'train_metadata.csv').exists(): train_meta=r/'Train'/'train_metadata.csv'; test_meta=r/'Test'/'test_metadata.csv'
        if train_meta.exists() and test_meta.exists():
            def imgdir(kind, ids):
                """Find the leaf directory containing the requested image IDs.

                Kaggle datasets commonly expose ``images/train_images`` or
                ``images/eval_images``; selecting the parent ``images`` would
                make otherwise valid metadata look unreadable.
                """
                ids = [str(x) for x in ids[:32]]
                preferred = [
                    r / "Train" / "images" / "train_images",
                    r / "Test" / "images" / "eval_images",
                    r / "images" / ("train_images" if kind == "train" else "eval_images"),
                    train_meta.parent / "images" / ("train_images" if kind == "train" else "eval_images"),
                ]
                candidates = [x for x in preferred if x.is_dir()]
                search_roots = [r, train_meta.parent]
                kaggle_input = Path("/kaggle/input")
                if kaggle_input.is_dir():
                    search_roots.append(kaggle_input)
                for base in search_roots:
                    if base.exists():
                        candidates.extend(x for x in base.rglob("*") if x.is_dir())
                scored = []
                for directory in set(candidates):
                    hits = sum((directory / image_id).is_file() for image_id in ids)
                    if hits:
                        scored.append((hits, len(str(directory)), directory))
                if not scored:
                    raise FileNotFoundError(
                        f"Could not locate {kind} image directory under {r}; "
                        f"sample IDs={ids[:3]}"
                    )
                scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
                return scored[0][2]
            tr=pd.read_csv(train_meta); te=pd.read_csv(test_meta)
            tr=tr.rename(columns={'sun_azimuth_angle':'azimuth'}); te=te.rename(columns={'sun_azimuth_angle':'azimuth'})
            tr.image_id=tr.image_id.astype(str); te.image_id=te.image_id.astype(str)
            return {'root':r,'train':tr,'test':te,'train_images':imgdir('train',tr.image_id.tolist()),'test_images':imgdir('test',te.image_id.tolist())}
    raise FileNotFoundError('Could not find train_metadata.csv and test_metadata.csv; set PAREIDOLIA_DATA_ROOT')

def load_image(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert('L'),dtype=np.float32)/255.

def rotate_image(image, azimuth):
    """Azimuth-normalize an image with the requested deterministic transform.

    PIL's positive angle is counter-clockwise, so ``-azimuth`` aligns the
    measured solar direction to the canonical horizontal frame.  ``expand``
    stays false to preserve the competition image geometry.
    """
    from PIL import Image
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(arr).rotate(
        -float(azimuth), resample=Image.Resampling.BILINEAR, expand=False
    ), dtype=np.float32) / 255.0

def image_sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def add_hashes(df, image_dir, verify=True):
    out=df.copy(); out['hash']=[image_sha256(Path(image_dir)/x) if verify else '' for x in out.image_id]; return out
