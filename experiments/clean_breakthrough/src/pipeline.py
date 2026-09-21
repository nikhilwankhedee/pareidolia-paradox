"""Small, shared experiment runner used by every clean-breakthrough notebook.

The runner deliberately separates image feature extraction from grouped OOF
classification.  It never accepts labels for test rows and writes all caches
under the caller-provided output directory.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from .dataset import load_image, rotate_image
from .azimuth import harmonic
from .classifiers import make
from .metrics import summarize
from .morphology import features as morphology_features
from .sfs import features as sfs_features
from .validation import splits


def _view_arrays(path, azimuth, kind):
    image = load_image(path)
    if kind == "tta":
        angles = (0.0, azimuth, azimuth - 15.0, azimuth + 15.0)
    elif kind in {"dino_multiview", "visual_multiview"}:
        angles = (0.0, azimuth, 90.0, -90.0)
    elif kind in {"dino_b14_azimuth", "dino_morphology", "dino_sfs"}:
        angles = (0.0, azimuth)
    else:
        angles = (0.0,)
    return [image if angle == 0 else rotate_image(image, angle) for angle in angles]


def _image_matrix(ids, image_dir, kind, azimuth=None):
    rows = []
    for image_id in ids:
        views = _view_arrays(Path(image_dir) / str(image_id),
                             azimuth[str(image_id)], kind)
        feats = []
        for image in views:
            if kind == "morphology":
                feats.append(morphology_features(image))
            elif kind == "sfs":
                feats.append(sfs_features(image, azimuth[str(image_id)]))
            else:
                feats.append(np.r_[morphology_features(image),
                                   sfs_features(image, azimuth[str(image_id)])])
        rows.append(np.mean(feats, axis=0))
    return np.vstack(rows).astype(np.float32)


def _foundation_matrix(ids, image_dir, kind, model, azimuth, preprocess=None):
    import torch
    from PIL import Image
    from torchvision import transforms
    device = next(model.parameters()).device
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    chunks = []
    for start in range(0, len(ids), 16):
        images = []
        for i in ids[start:start + 16]:
            views = _view_arrays(Path(image_dir) / str(i), azimuth[str(i)], kind)
            images.extend(views)
        pil_images = [Image.fromarray(np.clip(image * 255, 0, 255).astype("uint8"))
                      for image in images]
        if kind == "clip" and preprocess is not None:
            # Hugging Face CLIP processors return the pixel_values tensor
            # expected by CLIPModel; open_clip supplies a tensor transform.
            try:
                batch = preprocess(images=pil_images, return_tensors="pt")["pixel_values"]
            except (TypeError, KeyError):
                batch = torch.stack([preprocess(image) for image in pil_images])
        else:
            batch = torch.stack([transform(image) for image in pil_images])
        batch = batch.to(device)
        with torch.no_grad():
            if kind == "clip" and hasattr(model, "encode_image"):
                z = model.encode_image(batch)
            elif kind == "clip" and hasattr(model, "get_image_features"):
                z = model.get_image_features(pixel_values=batch)
            else:
                z = model(batch)
                if isinstance(z, (tuple, list)):
                    z = z[0]
                if z.ndim > 2:
                    z = z.mean(tuple(range(2, z.ndim)))
        z = z.detach().float().cpu().numpy()
        n_views = len(_view_arrays(Path(image_dir) / str(ids[start]),
                                    azimuth[str(ids[start])], kind))
        z = z.reshape(-1, n_views, z.shape[-1]).mean(axis=1)
        chunks.append(z)
    return np.vstack(chunks).astype(np.float32)


def extract_features(dataset, train, test, kind, cache_dir):
    """Extract/cache a feature matrix; optional foundation models fail loudly."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{kind}_features.npz"
    if kind in {"dino_b14", "dino_b14_azimuth", "dino_multiview",
                "dino_morphology", "dino_sfs", "clip", "pretrained_vision"}:
        # Probe the requested optional backend so the notebook reports the
        # real missing dependency/model error rather than silently substituting
        # a handcrafted feature for a foundation-model experiment.
        if kind.startswith("dino"):
            from .dino import load_dino
            model = load_dino("dinov2_vitb14")
        elif kind == "clip":
            from .clip import load_clip
            model, preprocess = load_clip()
        else:
            from .pretrained import load_encoder
            model = load_encoder()
            preprocess = None
            if hasattr(model, "fc"):
                import torch.nn as nn
                model.fc = nn.Identity()
        az_tr = dict(zip(train.image_id.astype(str), train.azimuth.astype(float)))
        az_te = dict(zip(test.image_id.astype(str), test.azimuth.astype(float)))
        xtr = _foundation_matrix(train.image_id, dataset["train_images"], kind, model, az_tr,
                                 preprocess)
        xte = _foundation_matrix(test.image_id, dataset["test_images"], kind, model, az_te,
                                 preprocess)
        if "azimuth" in kind:
            xtr = np.c_[xtr, harmonic(train.azimuth, 3)]
            xte = np.c_[xte, harmonic(test.azimuth, 3)]
        if kind in {"dino_morphology", "dino_sfs"}:
            suffix = "morphology" if kind.endswith("morphology") else "sfs"
            a_tr = dict(zip(train.image_id.astype(str), train.azimuth.astype(float)))
            a_te = dict(zip(test.image_id.astype(str), test.azimuth.astype(float)))
            xtr = np.c_[xtr, _image_matrix(train.image_id, dataset["train_images"], suffix, a_tr)]
            xte = np.c_[xte, _image_matrix(test.image_id, dataset["test_images"], suffix, a_te)]
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, train=xtr, test=xte,
                            train_id=train.image_id.to_numpy(),
                            test_id=test.image_id.to_numpy(), kind=kind)
        return xtr, xte, out
    az_tr = dict(zip(train.image_id.astype(str), train.azimuth.astype(float)))
    az_te = dict(zip(test.image_id.astype(str), test.azimuth.astype(float)))
    base_kind = "morphology" if kind == "morphology" else "sfs" if kind == "sfs" else kind
    xtr = _image_matrix(train.image_id, dataset["train_images"], base_kind, az_tr)
    xte = _image_matrix(test.image_id, dataset["test_images"], base_kind, az_te)
    if "azimuth" in kind or kind in {"tta", "dino_sfs"}:
        xtr = np.c_[xtr, harmonic(train.azimuth, 3)]
        xte = np.c_[xte, harmonic(test.azimuth, 3)]
    np.savez_compressed(out, train=xtr, test=xte,
                        train_id=train.image_id.to_numpy(),
                        test_id=test.image_id.to_numpy(), kind=kind)
    return xtr, xte, out


def grouped_oof(train, xtr, model="logistic", folds=5, seed=42):
    y = train.label.to_numpy(dtype=int)
    pred = np.zeros(len(train), dtype=np.float64)
    for ti, vi in splits(train, folds):
        pred[vi] = make(model, seed).fit(xtr[ti], y[ti]).predict_proba(xtr[vi])[:, 1]
    return pred, summarize(y, pred)


def run_experiment(dataset, train, test, kind, output_dir,
                   model="logistic", folds=5, seed=42):
    output_dir = Path(output_dir)
    xtr, xte, cache = extract_features(dataset, train, test, kind,
                                        output_dir / "cache")
    pred, metrics = grouped_oof(train, xtr, model, folds, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image_id": train.image_id, "prediction": pred,
                  "fold": train.fold, "experiment": kind}).to_csv(
                      output_dir / f"{kind}_oof.csv", index=False)
    (output_dir / f"{kind}_metrics.json").write_text(
        json.dumps(metrics, indent=2))
    return {"cache": str(cache), "oof": pred, "metrics": metrics,
            "test_features": xte}
