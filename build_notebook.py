"""
Builds experiment_3_signal_decomposition.ipynb — a Kaggle GPU notebook that
runs the entire Experiment 3 (Models A-E) with grouped-stratified CV.

Assembled with nbformat from cell source strings. Run:  python3 build_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python", "version": "3.12.3"},
}
cells = []

# ------------------------------------------------------------------
# Markdown: title + setup guide
# ------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(f"""# Experiment 3 — Leakage-Free Signal Decomposition

**The Pareidolia Paradox Dataset** · grouped 5-fold CV by exact image hash.

Runs five models to localize where predictive signal lives:
A) azimuth-only · B) raw image · C) raw image + azimuth ·
D) official-rotated image · E) official-rotated + azimuth.

- All five models use the **same** deterministic fold assignment (saved in the dataset).
- The test set is **never** touched.
- Threshold is first reported at 0.5, then a single *global* OOF-optimized threshold.
- CNNs use lazy loading + mixed precision + fixed seeds + checkpointing.

**Run All**: the notebook auto-detects the mounted dataset, validates it,
trains A–E on GPU, and writes everything to `/kaggle/working/experiment_3_outputs/`.
"""))

# ------------------------------------------------------------------
# Cell: imports + seeds + CUDA detect + auto path discovery
# ------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell("""import os, sys, time, json, zipfile, hashlib, math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report,
)

SEED = 42
N_FOLDS = 5

def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

set_seed(SEED)

# ---- CUDA ----
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('torch', torch.__version__)
print('device:', DEVICE)
if DEVICE == 'cuda':
    print('gpu:', torch.cuda.get_device_name(0))
else:
    print('WARNING: CUDA not available. CNNs B-E would be extremely slow on CPU.')
print('detected cpus:', os.cpu_count())
"""))

# ------------------------------------------------------------------
# Cell: auto-detect dataset path
# ------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell("""# ---- Auto-discover the mounted Kaggle dataset ----
# The dataset package contains: images/, train_metadata.csv,
# image_hashes.csv, fold_assignments.csv, README.txt
import os as _os

def _has_data(p):
    return _os.path.isfile(_os.path.join(str(p), "train_metadata.csv"))

DATA_ROOT = None
candidates = [
    Path('/kaggle/input/datasets/nikhilwankhedee/pd-dataset/pareidolia_experiment3_kaggle'),
    Path('/kaggle/input/pareidolia-experiment3-kaggle'),
    Path('/kaggle/input/pareidoliaexperiment3kaggle'),
    Path('/kaggle/input/pareidolia-experiment3'),
]
for c in candidates:
    if _has_data(c):
        DATA_ROOT = c; break

if DATA_ROOT is None and _os.path.isdir('/kaggle/input'):
    # walk the whole tree (followlinks traverses Kaggle's symlinked mounts).
    # /kaggle/input holds at most a handful of datasets, so a full walk is cheap.
    for root, dirs, files in _os.walk('/kaggle/input', followlinks=True):
        if 'train_metadata.csv' in files:
            DATA_ROOT = Path(root)
            break

if DATA_ROOT is None:
    # Diagnostics so we can see the actual mount layout.
    print('Could not auto-detect dataset. Contents of /kaggle/input:')
    for root, dirs, files in _os.walk('/kaggle/input', followlinks=True):
        depth = root[len('/kaggle/input/'):].count(_os.sep)
        print('  ' + '  ' * depth + root + ('/ ' + ', '.join(dirs) if depth < 3 else ''))
    raise RuntimeError(
        'Could not find train_metadata.csv under /kaggle/input. '
        'Attach the pareidolia_experiment3_kaggle dataset, then check the '
        'printed diagnostic for the actual mount path.')

IMG_DIR = DATA_ROOT / 'images'
META = DATA_ROOT / 'train_metadata.csv'
HASHES = DATA_ROOT / 'image_hashes.csv'
FOLDS = DATA_ROOT / 'fold_assignments.csv'
print('dataset root :', DATA_ROOT)
print('images       :', len(list(IMG_DIR.glob('*.png'))))
"""))

# ------------------------------------------------------------------
# Cell: load metadata, folds, hashes; validate structure
# ------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell("""meta = pd.read_csv(META)
fold_assign = pd.read_csv(FOLDS)
hash_tab = pd.read_csv(HASHES)

print('meta rows      :', len(meta))
print('fold rows      :', len(fold_assign))
print('hash rows      :', len(hash_tab))

# --- verify every metadata row maps to a fold and a hash ---
assert len(fold_assign) == len(meta), 'meta/fold row mismatch'
assert set(meta['image_id']) == set(fold_assign['image_id']), 'image_id mismatch'
assert set(meta['image_id']) == set(hash_tab['image_id']), 'hash id mismatch'

df = meta.merge(fold_assign[['image_id','fold']], on='image_id', validate='one_to_one')
df = df.merge(hash_tab, on='image_id', validate='one_to_one')
df = df.rename(columns={'sun_azimuth_angle':'azimuth'})
assert df['image_id'].nunique() == len(df) == 7854, 'expected 7854 rows'

print('merged frame   :', df.shape)
print('classes        :', df['label'].value_counts().to_dict())
"""))

# ------------------------------------------------------------------
# Cell: verify exact hash agreement between CSV and recomputed (sample)
# ------------------------------------------------------------------
cells.append(nbf.v4.new_code_markdown_cell() if False else nbf.v4.new_code_cell("""# --- Independently recompute hashes for a random subset and compare ---
def sha256_img(p):
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

rng = np.random.default_rng(SEED)
idx = rng.choice(len(df), size=min(300, len(df)), replace=False)
mism = 0
for i in idx:
    row = df.iloc[i]
    if sha256_img(IMG_DIR / row['image_id']) != row['hash']:
        mism += 1
print(f'recomputed hash mismatch over {len(idx)} sampled images: {mism}')
assert mism == 0, 'hash verification failed'
print('OK: package hashes match on-disk image bytes.')
"""))

# ------------------------------------------------------------------
# Cell: leakage-free fold validation
# ------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell("""# ---- Primary leakage check: no exact hash may span two folds ----
usage = defaultdict(set)
for h, f in zip(df['hash'], df['fold']):
    usage[h].add(int(f))
n_cross = sum(1 for u in usage.values() if len(u) > 1)
print('unique hashes/groups :', len(usage))
print('max folds per hash   :', max(len(u) for u in usage.values()))
print('cross-fold duplicate groups:', n_cross)
assert n_cross == 0, 'ABORT: hash crosses folds'

# group label composition
gc = df.groupby('hash')['label'].agg(lambda s: tuple(sorted(s)))
comp = gc.map(lambda t: 'mixed' if len(t) > 1 else str(t[0]))
print('group composition:', comp.value_counts().to_dict())

# ---- fold statistics ----
fold_stats = []
for f in range(N_FOLDS):
    sub = df[df['fold'] == f]
    c0 = int((sub['label'] == 0).sum()); c1 = int((sub['label'] == 1).sum())
    fold_stats.append({'fold': f, 'rows': len(sub),
                       'unique_hashes': sub['hash'].nunique(),
                       'class_0_count': c0, 'class_1_count': c1,
                       'class_0_prop': round(c0/len(sub),4),
                       'class_1_prop': round(c1/len(sub),4)})
fold_stats_df = pd.DataFrame(fold_stats)
print(fold_stats_df.to_string(index=False))
print('\\nGlobal class 0 proportion:', round((df['label']==0).mean(),4))
"""))

# ------------------------------------------------------------------
# Cell: metrics helpers
# ------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell("""def ba_at_threshold(y, p, t):
    return balanced_accuracy_score(y, (p >= t).astype(int))

def full_metrics(y, p, t):
    pred = (p >= t).astype(int)
    return {
        'balanced_accuracy': balanced_accuracy_score(y, pred),
        'recall_class_0': recall_score(y, pred, pos_label=0),
        'recall_class_1': recall_score(y, pred, pos_label=1),
        'roc_auc': roc_auc_score(y, p) if len(np.unique(y)) > 1 else float('nan'),
    }

def oof_optimal_threshold(y, p):
    best_t, best_ba = 0.5, ba_at_threshold(y, p, 0.5)
    for t in np.linspace(0.0, 1.0, 201):
        ba = ba_at_threshold(y, p, t)
        if ba > best_ba:
            best_ba, best_t = ba, t
    return best_t, best_ba
"""))

# ------------------------------------------------------------------
# Cell: shared small CNN (Model B/C/D/E architecture)
# ------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell("""# ---- Small deliberate CNN for signal localization (not SOTA) ----
# Input: 1x256x256 grayscale. 4 conv blocks + GAP + linear.
# use_azimuth: concatenate sin/cos azimuth to the pooled image feature.
class SmallCNN(nn.Module):
    def __init__(self, use_azimuth=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        feat = 128 + (2 if use_azimuth else 0)
        self.fc = nn.Linear(feat, 1)
        self.use_azimuth = use_azimuth
    def forward(self, x, az_sin=None, az_cos=None):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        if self.use_azimuth:
            a = torch.stack([az_sin, az_cos], dim=1)
            x = torch.cat([x, a], dim=1)
        return self.fc(x).squeeze(1)

def count_params(m):
    return sum(p.numel() for p in m.parameters())
print('params B/D:', count_params(SmallCNN(use_azimuth=False)))
print('params C/E:', count_params(SmallCNN(use_azimuth=True)))
"""))

# ------------------------------------------------------------------
# Cell: dataset with lazy loading + official rotation preprocessing
# ------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell("""# ---- Dataset: lazy grayscale loading; optional official rotation ----
# rotate = image.rotate(-sun_azimuth_angle), PIL BILINEAR, expand=False,
# i.e. the exact documented preprocessing from the rotation audit.
class PareidoliaDS(Dataset):
    def __init__(self, frame, img_dir, rotate=False):
        self.frame = frame.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.rotate = rotate  # D/E: True ; B/C: False
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, i):
        r = self.frame.iloc[i]
        img = Image.open(self.img_dir / r['image_id']).convert('L')
        if self.rotate:
            img = img.rotate(-float(r['azimuth']), resample=Image.BILINEAR, expand=False)
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(x).unsqueeze(0)  # 1 x H x W
        az = float(r['azimuth'])
        az_sin = math.sin(math.radians(az)); az_cos = math.cos(math.radians(az))
        y = float(r['label'])
        return x, torch.tensor([az_sin, az_cos], dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
"""))

# ------------------------------------------------------------------
# Cell: training function (one fold) with identical protocol B-E
# ------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell("""def train_fold(model, tr_df, va_df, img_dir, rotate, cfg):
    '''Train one fold for a CNN (B/C/D/E). Identical protocol for all.'''
    set_seed(cfg['seed'])
    tr_ds = PareidoliaDS(tr_df, img_dir, rotate=rotate)
    va_ds = PareidoliaDS(va_df, img_dir, rotate=rotate)
    tr_loader = DataLoader(tr_ds, batch_size=cfg['batch_size'], shuffle=True,
                           num_workers=0, pin_memory=(DEVICE=='cuda'), drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=cfg['batch_size'], shuffle=False,
                           num_workers=0, pin_memory=(DEVICE=='cuda'))

    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs'])
    lossf = nn.BCEWithLogitsLoss()   # plain BCE, no class weighting (comparability)
    use_amp = (DEVICE == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_ba, best_state = -1.0, None
    tstart = time.time()
    for ep in range(cfg['epochs']):
        model.train()
        for x, az, y in tr_loader:
            x, az, y = x.to(DEVICE), az.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                if model.use_azimuth:
                    logit = model(x, az[:,0], az[:,1])
                else:
                    logit = model(x)
                loss = lossf(logit, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        # validate
        model.eval()
        p, yv = [], []
        with torch.no_grad():
            for x, az, y in va_loader:
                x, az = x.to(DEVICE), az.to(DEVICE)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    if model.use_azimuth:
                        lg = model(x, az[:,0], az[:,1])
                    else:
                        lg = model(x)
                p.append(torch.sigmoid(lg.float()).cpu().numpy())
                yv.append(y.numpy())
        p = np.concatenate(p); yv = np.concatenate(yv)
        ba = balanced_accuracy_score(yv == 1, (p >= 0.5).astype(int))
        if ba > best_ba:
            best_ba = ba
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"    epoch {ep+1}/{cfg['epochs']} val_BA={ba:.4f} "
              f"elapsed={time.time()-tstart:.0f}s", flush=True)
    return best_state, best_ba, len(tr_df), len(va_df)
"""))

# ------------------------------------------------------------------
# Cell: OOF inference for a model
# ------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell("""def oof_predict(model, va_df, img_dir, rotate):
    '''Return sigmoid probabilities for validation samples at best checkpoint.'''
    ds = PareidoliaDS(va_df, img_dir, rotate=rotate)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    model.eval()
    ps, idxs = [], []
    with torch.no_grad():
        for x, az, _ in loader:
            x, az = x.to(DEVICE), az.to(DEVICE)
            if model.use_azimuth:
                lg = model(x, az[:,0], az[:,1])
            else:
                lg = model(x)
            ps.append(torch.sigmoid(lg.float()).cpu().numpy())
    return np.concatenate(ps)
"""))

# ------------------------------------------------------------------
# Cell: run CNNs B,C,D,E across all folds (the main GPU workload)
# ------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell("""CFG = {
    'seed': SEED, 'lr': 1e-3, 'weight_decay': 1e-4,
    'batch_size': 32, 'epochs': 12, 'label_weighting': 'none',
}

MODEL_SPECS = [
    ('B_raw_image',          dict(rotate=False, use_azimuth=False)),
    ('C_raw_image_azimuth',  dict(rotate=False, use_azimuth=True)),
    ('D_rotated_image',      dict(rotate=True,  use_azimuth=False)),
    ('E_rotated_azimuth',    dict(rotate=True,  use_azimuth=True)),
]

# store OOF probabilities (rows aligned to df order via index)
oof_maps = {}          # model_name -> np array (len=df) with NaN for train rows
train_times = {}       # model_name -> {fold: seconds}
for name, spec in MODEL_SPECS:
    print(f'\\n==== {name} ====')
    oof = np.full(len(df), np.nan)
    times = {}
    for f in range(N_FOLDS):
        tr_idx = np.where(df['fold'].values != f)[0]
        va_idx = np.where(df['fold'].values == f)[0]
        tr_df = df.iloc[tr_idx]
        va_df = df.iloc[va_idx]
        model = SmallCNN(use_azimuth=spec['use_azimuth'])
        t0 = time.time()
        print(f'  fold {f}/{N_FOLDS}: training (train={len(tr_idx)} val={len(va_idx)})...',
              flush=True)
        best_state, best_ba, ntr, nva = train_fold(model, tr_df, va_df,
                                                   IMG_DIR, spec['rotate'], CFG)
        dt = time.time() - t0
        times[f] = dt
        model.load_state_dict(best_state)
        oof[va_idx] = oof_predict(model, va_df, IMG_DIR, spec['rotate'])
        print(f'  fold {f}: best_val_BA={best_ba:.4f} time={dt:.0f}s '
              f'(train={ntr} val={nva})')
    oof_maps[name] = oof
    train_times[name] = times
    Path('/kaggle/working/experiment_3_outputs/model_checkpoints').mkdir(parents=True, exist_ok=True)
    torch.save({'oof': oof, 'spec': spec}, f'/kaggle/working/experiment_3_outputs/model_checkpoints/{name}_oof.pt')
print('\\nAll CNN models trained.')
"""))

# ------------------------------------------------------------------
# Cell: Model A (azimuth-only, CPU) + OOF
# ------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell("""# --- Model A: azimuth-only sin/cos logistic regression ---
theta = np.deg2rad(df['azimuth'].values)
X_az = np.stack([np.sin(theta), np.cos(theta)], axis=1)
oof_a = np.full(len(df), np.nan)
times_a = {}
for f in range(N_FOLDS):
    tr = df['fold'].values != f
    va = df['fold'].values == f
    t0 = time.time()
    clf = LogisticRegression(max_iter=2000, random_state=SEED)
    clf.fit(X_az[tr], df['label'].values[tr])
    oof_a[va] = clf.predict_proba(X_az[va])[:, 1]
    times_a[f] = time.time() - t0
oof_maps['A_azimuth'] = oof_a
train_times['A_azimuth'] = times_a
print('Model A OOF done.')
"""))

# ------------------------------------------------------------------
# Cell: assemble OOF table + verify completeness
# ------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell("""# ---- Assemble the OOF CSV: one row per training image ----
oof_out = pd.DataFrame({
    'image_id': df['image_id'],
    'hash': df['hash'],
    'fold': df['fold'],
    'label': df['label'],
    'azimuth': df['azimuth'],
    'prob_A_azimuth': oof_maps['A_azimuth'],
    'prob_B_raw': oof_maps['B_raw_image'],
    'prob_C_raw_azimuth': oof_maps['C_raw_image_azimuth'],
    'prob_D_rotated': oof_maps['D_rotated_image'],
    'prob_E_rotated_azimuth': oof_maps['E_rotated_azimuth'],
})
# verify completeness
assert len(oof_out) == 7854
assert oof_out['image_id'].nunique() == 7854
for c in ['prob_A_azimuth','prob_B_raw','prob_C_raw_azimuth',
          'prob_D_rotated','prob_E_rotated_azimuth']:
    assert oof_out[c].notna().all(), f'{c} has NaN (missing prediction)'
print('OOF table complete:', oof_out.shape)
print(oof_out.head(3).to_string())
"""))

# ------------------------------------------------------------------
# Cell: compute per-fold + aggregate metrics for every model
# ------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell("""# ---- Per-fold + aggregate metrics for every model ----
MODEL_ORDER = ['A_azimuth','B_raw_image','C_raw_image_azimuth',
               'D_rotated_image','E_rotated_azimuth']

y = df['label'].values
all_rows = []       # model x fold
agg_rows = []
for name in MODEL_ORDER:
    p = oof_maps[name]
    per_fold = []
    for f in range(N_FOLDS):
        va = df['fold'].values == f
        m = full_metrics(y[va], p[va], 0.5)
        per_fold.append(m)
        all_rows.append({'model': name, 'fold': f,
                         'n': int(va.sum()),
                         'class_0_count': int((y[va]==0).sum()),
                         'class_1_count': int((y[va]==1).sum()),
                         'balanced_accuracy': m['balanced_accuracy'],
                         'recall_class_0': m['recall_class_0'],
                         'recall_class_1': m['recall_class_1'],
                         'roc_auc': m['roc_auc'],
                         'threshold': 0.5,
                         'training_time_seconds': train_times[name].get(f, np.nan)})
    bas = [m['balanced_accuracy'] for m in per_fold]
    best_t, best_ba = oof_optimal_threshold(y, p)
    agg_rows.append({
        'model': name,
        'mean_BA': float(np.mean(bas)),
        'std_BA': float(np.std(bas)),
        'OOF_BA': ba_at_threshold(y, p, 0.5),
        'OOF_recall_class_0': recall_score(y, (p>=0.5).astype(int), pos_label=0),
        'OOF_recall_class_1': recall_score(y, (p>=0.5).astype(int), pos_label=1),
        'OOF_ROC_AUC': roc_auc_score(y, p),
        'optimal_OOF_threshold': float(best_t),
        'optimal_OOF_BA': float(best_ba),
        'training_time_total_s': float(sum(train_times[name].values())),
    })

res_df = pd.DataFrame(all_rows)
agg_df = pd.DataFrame(agg_rows)
agg_df = agg_df.sort_values('OOF_BA', ascending=False)
print(agg_df[['model','mean_BA','std_BA','OOF_BA','OOF_recall_class_0',
              'OOF_recall_class_1','OOF_ROC_AUC','optimal_OOF_threshold',
              'optimal_OOF_BA']].to_string(index=False))
"""))

# ------------------------------------------------------------------
# Cell: confusion matrices (t=0.5 and OOF-optimal) for each model
# ------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell("""# ---- Confusion matrices at t=0.5 and at OOF-optimal threshold ----
conf_parts = []
for name in MODEL_ORDER:
    p = oof_maps[name]
    t_op = agg_df.loc[agg_df['model']==name, 'optimal_OOF_threshold'].iloc[0]
    for t, tag in [(0.5,'t0.5'), (t_op, 't_opt')]:
        cm = confusion_matrix(y, (p >= t).astype(int))
        conf_parts.append({'model': name, 'threshold': f'{t:.3f}', 'tag': tag,
                           'TN': int(cm[0,0]), 'FP': int(cm[0,1]),
                           'FN': int(cm[1,0]), 'TP': int(cm[1,1])})
        print(f'{name} @ t={t:.3f}  [[{cm[0,0]} {cm[0,1]}]\\n            [{cm[1,0]} {cm[1,1]}]]')
conf_df = pd.DataFrame(conf_parts)
"""))

# ------------------------------------------------------------------
# Cell: azimuth-bin diagnostics
# ------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell("""# ---- Azimuth-bin diagnostics (diagnostic only, not optimization) ----
bins = [0,45,90,135,180,225,270,315,360]
labels_az = ['0-45','45-90','90-135','135-180','180-225','225-270','270-315','315-360']
bin_idx = pd.cut(df['azimuth'], bins=bins, labels=labels_az, right=False)
az_rows = []
for name in MODEL_ORDER:
    p = oof_maps[name]
    for b in labels_az:
        sel = (bin_idx == b).values
        if sel.sum() == 0:
            continue
        c0 = (y[sel]==0).sum(); c1 = (y[sel]==1).sum()
        m = full_metrics(y[sel], p[sel], 0.5)
        az_rows.append({'model': name, 'azimuth_bin': b,
                        'sample_count': int(sel.sum()),
                        'class_0_prop': round(c0/sel.sum(),3),
                        'BA': m['balanced_accuracy'],
                        'recall_class_0': m['recall_class_0'],
                        'recall_class_1': m['recall_class_1']})
az_df = pd.DataFrame(az_rows)
print(az_df.to_string(index=False))
"""))

# ------------------------------------------------------------------
# Cell: duplicate-group diagnostics
# ------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell("""# ---- Duplicate-group diagnostics ----
# A sample is in a duplicate group if its hash appears more than once.
gp = df.groupby('hash')['label'].transform('size')
dup_mask = gp > 1
sing = ~dup_mask
mixed = df['hash'].isin(gc[gc.map(len)>1].index).values  # mixed-label dup groups

diag = {'total': int(len(df)),
        'singleton_samples': int(sing.sum()),
        'duplicate_group_samples': int(dup.sum()),
        'mixed_label_dup_samples': int(mixed.sum())}
print(diag)

# Full (leakage-free) OOF score vs singleton-only OOF score
print('\\nFull vs singleton-only OOF BA (t=0.5):')
for name in MODEL_ORDER:
    p = oof_maps[name]
    full = ba_at_threshold(y, p, 0.5)
    s_only = ba_at_threshold(y[sing], p[sing], 0.5) if sing.sum()>0 else float('nan')
    print(f'  {name:22s} full={full:.4f}  singleton-only={s_only:.4f}')
"""))

# ------------------------------------------------------------------
# Cell: save output CSVs + config
# ------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell("""# ---- Save CSV outputs ----
OUT = Path('/kaggle/working/experiment_3_outputs')
OUT.mkdir(parents=True, exist_ok=True)

oof_out.to_csv(OUT / 'signal_decomposition_oof.csv', index=False)
fold_stats_df.to_csv(OUT / 'fold_statistics.csv', index=False)
df[['image_id','hash','label','azimuth','fold']].to_csv(OUT / 'fold_assignments.csv', index=False)
res_df.to_csv(OUT / 'experiment_results.csv', index=False)
agg_df.to_csv(OUT / 'aggregate_results.csv', index=False)
conf_df.to_csv(OUT / 'confusion_matrices.csv', index=False)
az_df.to_csv(OUT / 'azimuth_bin_diagnostics.csv', index=False)

cfg = {
    'seed': SEED, 'n_folds': N_FOLDS,
    'cnn': {'architecture': 'SmallCNN 4x(Conv3x3-BN-ReLU-MP) + GAP + Linear',
            'params_B/D': count_params(SmallCNN(False)),
            'params_C/E': count_params(SmallCNN(True)),
            'optimizer': 'AdamW', 'lr': 1e-3, 'weight_decay': 1e-4,
            'batch_size': CFG['batch_size'], 'epochs': CFG['epochs'],
            'loss': 'BCEWithLogitsLoss (no class weighting)',
            'scheduler': 'CosineAnnealingLR', 'amp': DEVICE=='cuda',
            'rotation': 'PIL rotate(-azimuth) BILINEAR expand=False (D/E)',
            'selection': 'best validation BA per fold'},
}
with open(OUT / 'training_config.json', 'w') as f:
    json.dump(cfg, f, indent=2)
print('Saved CSVs + config to', OUT)
"""))

# ------------------------------------------------------------------
# Cell: generate experiment report markdown
# ------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell("""# ---- Generate experiment_report.md ----
from datetime import datetime
def md_table(rows, header):
    s = '| ' + ' | '.join(header) + ' |\\n'
    s += '|' + '|'.join(['---']*len(header)) + '|\\n'
    for r in rows:
        s += '| ' + ' | '.join(str(x) for x in r) + ' |\\n'
    return s

lines = []
lines.append('# Experiment 3 — Leakage-Free Signal Decomposition')
lines.append('')
lines.append(f'Generated: {datetime.utcnow().isoformat()}Z · device={DEVICE} · '\
             f'torch {torch.__version__} · seed={SEED}')
lines.append('')
lines.append('## 1. Experimental setup')
lines.append('')
lines.append('**Dataset:** Pareidolia Paradox train split (7,854 images, 256x256 grayscale lunar; '\
             'class 0=Depth (2,854), class 1=Rise (5,000)). Test set excluded entirely.')
lines.append('')
lines.append('**Grouping:** each exact sha256 image hash is one group. 6,396 unique groups; '\
             'of which 1,458 are 2-image duplicate groups, ALL mixed-label (0 and 1).')
lines.append('')
lines.append('**Fold construction:** singletons stratified by own label via StratifiedKFold(5) '\
             '(seed=42); mixed (0,1) groups distributed round-robin after seeded shuffle. '\
             'Assignment is fixed and shared by all five models. Verified: max folds per hash = 1, '\
             'cross-fold duplicate groups = 0.')
lines.append('')
lines.append('### Fold statistics')
lines.append('')
lines.append(md_table(
    [[r['fold'], r['rows'], r['unique_hashes'], r['class_0_count'], r['class_1_count'],
      r['class_0_prop'], r['class_1_prop']] for _, r in fold_stats_df.iterrows()],
    ['fold','rows','unique_hashes','class_0_count','class_1_count','class_0_prop','class_1_prop']))
lines.append('')
lines.append('**Models:** A) sin/cos azimuth + logistic regression. B) raw small CNN. '\
             'C) raw CNN + sin/cos azimuth concat. D) official-rotated image CNN. '\
             'E) rotated CNN + sin/cos azimuth. B-E share architecture (SmallCNN), seed, optim '\
             '(AdamW 1e-3, wd 1e-4), scheduler (CosAnnealing), loss (plain BCE), batch 32, '\
             'E' + chr(37) + 's=' + str(CFG['epochs']) + ', AMP=' + str(DEVICE=='cuda') + '.')
lines.append('')
lines.append('**Rotation (D/E):** PIL `image.rotate(-sun_azimuth_angle)`, BILINEAR, expand=False, '\
             'deterministic (official convention A).')
lines.append('')

# Model results
lines.append('## Model results (aggregate / OOF)')
lines.append('')
lines.append(md_table(
    [[r['model'], f"{r['mean_BA']:.4f}", f"{r['std_BA']:.4f}", f"{r['OOF_BA']:.4f}",
      f"{r['OOF_recall_class_0']:.4f}", f"{r['OOF_recall_class_1']:.4f}",
      f"{r['OOF_ROC_AUC']:.4f}", f"{r['optimal_OOF_threshold']:.3f}",
      f"{r['optimal_OOF_BA']:.4f}"] for _, r in agg_df.iterrows()],
    ['model','mean_BA','std_BA','OOF_BA','OOF_R0','OOF_R1','OOF_AUC','opt_thresh','opt_BA']))
lines.append('')
lines.append('Threshold policy: results reported at t=0.5, then a single global OOF-optimized '\
             'threshold (no per-fold tuning reported as untouched).')
lines.append('')

# Confusion matrices
lines.append('## Confusion matrices ([[0,0],[1,1]] rowwise; rows=class0,class1)')
lines.append('')
for _, r in conf_df.iterrows():
    lines.append(f"- `{r['model']}` @ t={r['threshold']}: [[{r['TN']} {r['FP']}], [{r['FN']} {r['TP']}]]")
lines.append('')

# Azimuth bins
lines.append('## Azimuth-bin diagnostics (BA at t=0.5)')
lines.append('')
lines.append(md_table(
    [[r['model'], r['azimuth_bin'], r['sample_count'], f"{r['class_0_prop']:.3f}",
      f"{r['BA']:.3f}", f"{r['recall_class_0']:.3f}", f"{r['recall_class_1']:.3f}"]
     for _, r in az_df.iterrows()],
    ['model','azimuth_bin','n','c0_prop','BA','R0','R1']))
lines.append('')

# Duplicate-group diagnostics
lines.append('## Duplicate-group diagnostics')
lines.append('')
lines.append(f"- total={diag['total']}, singleton={diag['singleton_samples']}, "
             f"duplicate-group={diag['duplicate_group_samples']}, "
             f"mixed-label-dup={diag['mixed_label_dup_samples']}")
lines.append('All 1,458 duplicate groups are mixed-label (identical pixels labeled 0 and 1).')
lines.append('')
lines.append('Full vs singleton-only OOF BA (t=0.5):')
lines.append('')
for name in MODEL_ORDER:
    p = oof_maps[name]
    full = ba_at_threshold(y, p, 0.5)
    s_only = ba_at_threshold(y[sing], p[sing], 0.5) if sing.sum()>0 else float('nan')
    lines.append(f"- `{name}`: full={full:.4f}, singleton-only={s_only:.4f}")
lines.append('')

# Scientific interpretation placeholder (filled by lead on receiving results)
lines.append('## Scientific interpretation (filled after inspection of results)')
lines.append('')
lines.append('_(The narrative answers to the six scientific questions, and the single next '
             'experiment, are finalized by the lead researcher once these measurements are '
             'reviewed. The numeric evidence above is the raw signal-localization data.)_')
lines.append('')

with open(OUT / 'experiment_report.md', 'w') as f:
    f.write('\\n'.join(lines))
print('Wrote experiment_report.md')
"""))

# ------------------------------------------------------------------
# Cell: zip outputs and print summary
# ------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell("""# ---- Compress the outputs ----
out_zip = Path('/kaggle/working/experiment_3_results.zip')
with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.rglob('*')):
        if p.is_file():
            z.write(p, arcname=str(p.relative_to(OUT)))

print('=== DONE ===')
print('Outputs in', OUT)
for p in sorted(OUT.rglob('*')):
    if p.is_file():
        print(f'  {p.name}  {p.stat().st_size:,} B')
print()
print('OOF BA ranking (aggregate):')
for _, r in agg_df.iterrows():
    print(f"  {r['model']:22s} OOF BA={r['OOF_BA']:.4f} ± {r['std_BA']:.4f}  AUC={r['OOF_ROC_AUC']:.4f}")
print('Download /kaggle/working/experiment_3_results.zip and send back.')
"""))

nb["cells"] = cells

# ---- Inject a "ran OK" completion marker into every CODE cell ----
# The user wants each cell to print when it has run, so the notebook is easy
# to follow and any failed cell is immediately obvious. Labels are derived
# from each code cell's leading comment or a short explicit name.
def _cell_marker(source_lines):
    for ln in source_lines:
        s = ln.strip()
        if s.startswith('#'):
            # prefer an explicit "# Cell: ..." header else the first '#....'
            return s.lstrip('#').strip()
    return 'code'

# Explicit names are clearer than the long comment strings; map cell index -> short label.
EXPLICIT = {
    1: 'setup / seeds / CUDA detect',
    2: 'locate + mount dataset',
    3: 'load metadata + folds + hashes',
    4: 'recompute-hash verification',
    5: 'fold validation (leakage check)',
    6: 'metrics helpers',
    7: 'define SmallCNN',
    8: 'dataset + rotation preprocessing',
    9: 'training function (train_fold)',
    10: 'OOF inference (oof_predict)',
    11: 'train CNNs B,C,D,E',
    12: 'Model A (azimuth-only)',
    13: 'assemble OOF table',
    14: 'per-fold + aggregate metrics',
    15: 'confusion matrices',
    16: 'azimuth-bin diagnostics',
    17: 'duplicate-group diagnostics',
    18: 'save output CSVs + config',
    19: 'generate experiment_report.md',
    20: 'compress outputs + summary',
}

injected = 0
for idx, c in enumerate(cells):
    if c.cell_type == 'code':
        lines = c.source.rstrip().split('\n')
        label = EXPLICIT.get(idx, _cell_marker(lines))
        c.source = c.source.rstrip() + f'\nprint(">>> ran OK [' + str(idx) + ']: ' + label + '")\n'
        injected += 1

nbf.write(nb, "notebooks/experiment_3_signal_decomposition.ipynb")
print(f"Wrote notebooks/experiment_3_signal_decomposition.ipynb with {len(cells)} cells "
      f"({injected} code cells now print a completion marker)")
