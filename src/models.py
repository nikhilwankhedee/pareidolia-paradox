"""Model wrappers: azimuth LGBM, logistic regression, small LGBM."""
import numpy as np

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def make_az_model(n_estimators=120, learning_rate=0.05, num_leaves=31,
                  max_depth=3, subsample=0.8, colsample_bytree=0.8, seed=42):
    return lgb.LGBMClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
        max_depth=max_depth, subsample=subsample, subsample_freq=1,
        colsample_bytree=colsample_bytree, random_state=seed, verbose=-1)


def fit_az_predict(y, az_fit, az_query, seed=42):
    """Fit azimuth-only LGBM (order-1 harmonic already applied by caller)."""
    clf = make_az_model(seed=seed)
    clf.fit(np.asarray(az_fit, dtype=float), np.asarray(y, dtype=int))
    return clf.predict_proba(np.asarray(az_query, dtype=float))[:, 1]


def fit_lr(X, y, C=1.0, seed=42):
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    model = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    model.fit(Xs, y)
    return lambda Xq: model.predict_proba(sc.transform(Xq))[:, 1]


def fit_lgb(X, y, seed=42, n_estimators=200, num_leaves=15):
    model = lgb.LGBMClassifier(n_estimators=n_estimators, num_leaves=num_leaves,
                               learning_rate=0.05, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, random_state=seed, verbose=-1)
    model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int))
    return lambda Xq: model.predict_proba(np.asarray(Xq, dtype=float))[:, 1]