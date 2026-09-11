# -*- coding: utf-8 -*-
"""E64: 外部元评审补充计算 — ①仅分数特征基线 ②HGB种子方差 ③家族Wilson CI."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
CACHE = os.path.join(SRC, '_cache')

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

rows = json.load(open(os.path.join(CACHE, 'e3_windows.json'),
                      encoding='utf-8'))
feats = [c for c in rows[0] if c[0] in 'gcasd' and c[1] == '_']
s_feats = [f for f in feats if f.startswith('s_')]
X = np.array([[r.get(f, np.nan) for f in feats] for r in rows])
Xs = np.array([[r.get(f, np.nan) for f in s_feats] for r in rows])
y = np.array([r['label'] for r in rows])
ds = np.array([r['dataset'] for r in rows])
s_idx = [feats.index(f) for f in s_feats]
print(f'n={len(rows)}, 特征{len(feats)}个, 其中分数特征{len(s_feats)}个: {s_feats}')


def lodo_auroc(Xm, seed=0):
    preds = np.zeros(len(y))
    for d in np.unique(ds):
        m_tr, m_te = ds != d, ds == d
        clf = HistGradientBoostingClassifier(random_state=seed)
        clf.fit(Xm[m_tr], y[m_tr])
        preds[m_te] = clf.predict_proba(Xm[m_te])[:, 1]
    return roc_auc_score(y, preds)


# ① 仅分数特征
auroc_s = lodo_auroc(Xs)
print(f'\n仅分数特征(s_*) LODO AUROC = {auroc_s:.3f}')

# ② HGB 种子方差(全特征)
aurocs = [lodo_auroc(X, seed=s) for s in range(5)]
print(f'全特征 LODO AUROC × 5种子: {[f"{a:.3f}" for a in aurocs]} '
      f'(均值{np.mean(aurocs):.3f}, 范围[{min(aurocs):.3f},{max(aurocs):.3f}])')

# ③ 家族等权均值的 Wilson 区间(来自 a1 发布工件)
from math import sqrt
a = json.load(open(os.path.join(CACHE, 'a1_analysis.json'),
                   encoding='utf-8'))
bm = a['quadrant']['by_method']
fams = {k: (v['stable_wrong'], v['n']) for k, v in bm.items()
        if k != 'Random'}


def wilson(p, n, z=1.96):
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - hw, c + hw


print('\n家族 Wilson 95% CI (发布工件):')
for k, (p, n) in sorted(fams.items(), key=lambda x: -x[1][0]):
    lo, hi = wilson(p, n)
    print(f'  {k}: {p:.3f} [{lo:.3f}, {hi:.3f}] (n={n})')

out = dict(score_only_auroc=float(auroc_s),
           hgb_seed_aurocs=[float(a) for a in aurocs],
           family_shares={k: dict(p=float(v[0]), n=v[1]) for k, v in fams.items()})
json.dump(out, open(os.path.join(CACHE, 'e64_review_addendum.json'), 'w'))
