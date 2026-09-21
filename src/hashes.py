"""Byte-hash forensics: exact duplicates, overlaps, intra pairs, novel set."""
import hashlib
from pathlib import Path


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def hash_images(img_dir, id_set):
    """Return (hash->[ids], missing_ids)."""
    out = {}
    if img_dir is None:
        return {}, set(id_set)
    for p in sorted(Path(img_dir).glob("*")):
        if p.name not in id_set:
            continue
        out.setdefault(sha256_file(p), []).append(p.name)
    done = set(i for ids in out.values() for i in ids)
    return out, set(id_set) - done


def partition(tr_hash, te_hash, train_ids, test_ids):
    """Return dict with regimes and group tables.

    - train_dup_groups: hash -> [ids] for train-only dups (size>=2)
    - overlaps: (hash, train_id, test_id) rows for hashes shared train<->test
    - intra: hash -> [ids] for test-internal dups (size>=2, hash not in train)
    - novel: test ids whose hash has exactly one test member and no train twin
    """
    tr_dup = {h: ids for h, ids in tr_hash.items() if len(ids) >= 2}
    ov_hash = {h for h in te_hash if h in tr_hash}
    intra = {h: ids for h, ids in te_hash.items()
             if h not in tr_hash and len(ids) >= 2}
    is_ov = set()
    for h in ov_hash:
        for t in te_hash[h]:
            is_ov.add(t)
    is_intra = set()
    for h, ids in intra.items():
        for t in ids:
            is_intra.add(t)
    novel = [t for t in test_ids if t not in is_ov and t not in is_intra]

    ov_list = []
    for h in sorted(ov_hash):
        for ti in sorted(tr_hash[h]):
            for te in sorted(te_hash[h]):
                ov_list.append((h, ti, te))
    return {
        "train_dup_groups": tr_dup,
        "overlap_hash": ov_hash,
        "intra_groups": intra,
        "novel": novel,
        "is_overlap": is_ov,
        "is_intra": is_intra,
        "overlap_rows": ov_list,
    }


def count_report(part, n_overlap, n_intra, n_novel, n_train_dup_groups):
    return {
        "train_dup_groups": len(n_train_dup_groups) if isinstance(n_train_dup_groups, dict) else n_train_dup_groups,
        "overlap_test_images": n_overlap,
        "intra_test_images": n_intra,
        "intra_pairs": (n_intra // 2) if n_intra % 2 == 0 else -1,
        "novel_test_images": n_novel,
        "sum_check": n_overlap + n_intra + n_novel,
    }