"""Exact-image hashing helpers.

The Pareidolia competition images are PNGs that are byte-identical across many
rows. Exact duplication is checked via the SHA-256 of the raw file bytes, which
is unambiguous and fast.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict


def sha256_bytes(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's raw bytes."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def hash_table(image_ids, img_dir) -> Dict[str, str]:
    """Map image_id -> exact sha256 hex digest of its PNG bytes.

    Parameters
    ----------
    image_ids : iterable of str
        File names (image_id) to hash.
    img_dir : path-like
        Directory containing the images.
    """
    img_dir = Path(img_dir)
    table = {}
    for img in image_ids:
        table[img] = sha256_bytes(img_dir / img)
    return table


def group_label_composition(frame: "pd.DataFrame", hash_col: str, label_col: str) -> Dict:
    """Summarize each hash-group's label composition.

    Returns {hash -> ('0' | '1' | 'mixed')} where 'mixed' means the group
    contains both class-0 and class-1 labels (i.e. pixel-identical images
    that are labeled differently).
    """
    gc = frame.groupby(hash_col)[label_col].agg(lambda s: tuple(sorted(s)))
    return gc.map(lambda t: "mixed" if len(t) > 1 else str(t[0])).to_dict()
