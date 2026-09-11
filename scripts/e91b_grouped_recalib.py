# -*- coding: utf-8 -*-
"""E91b: 跨族重校准, 按窗口索引分组的无泄漏版 (e91 的方法学加固).

采样单位 = 窗口索引 (该窗口的全部方法行同侧), 排除同窗跨方法泄漏;
段连续性下窗口级分组也近似段级隔离。n 报告为标注窗口实例数(行)。
-> _cache/e91_cross_recalib_grouped.json
"""
import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '..', '_cache')

scm = json.load(open(os.path.join(CACHE, 'e3_windows.json'), encoding='utf-8'))
wad = json.load(open(os.path.join(CACHE, 'e85_wadi_rows.json'),
                     encoding='utf-8'))
feats = [c for c in wad[0] if len(c) > 2 and c[0] in 'gcasd' and c[1] == '_']

Xs = np.array([[r.get(f, np.nan) for f in feats] for r in scm])
ys = np.array([r['label'] for r in scm])
widx = np.array([r['window'] for r in wad])
yw = np.array([r['label'] for r in wad])
Xw = np.array([[r.get(f, np.nan) for f in feats] for r in wad])
uniq = np.unique(widx)
pos_frac_by_win = {w: yw[widx == w].mean() for w in uniq}
print('windows', len(uniq), 'rows', len(yw), flush=True)

out = {}
for n_rows in (0, 50, 100, 200, 400):
    aucs, ns = [], []
    for rep in range(5):
        rng = np.random.default_rng(2000 + rep)
        if n_rows > 0:
            k = max(1, n_rows // 9)          # 每窗约 9 行(方法数)
            # 按 window 正例率分层: 一半取高正例率窗, 一半低
            order = sorted(uniq, key=lambda w: (pos_frac_by_win[w], w))
            half = len(order) // 2
            pool_hi, pool_lo = order[half:], order[:half]
            sel = np.concatenate([
                rng.choice(pool_hi, k // 2, replace=False),
                rng.choice(pool_lo, k - k // 2, replace=False)]) \
                if k <= half * 2 else rng.choice(uniq, k, replace=False)
            tr_mask = np.isin(widx, sel)
            ns.append(int(tr_mask.sum()))
        else:
            tr_mask = np.zeros(len(yw), dtype=bool)
            ns.append(0)
        te_mask = ~tr_mask
        clf = HistGradientBoostingClassifier(random_state=rep)
        if n_rows > 0:
            clf.fit(np.vstack([Xs, Xw[tr_mask]]),
                    np.concatenate([ys, yw[tr_mask]]))
        else:
            clf.fit(Xs, ys)
        p = clf.predict_proba(Xw[te_mask])[:, 1]
        if len(set(yw[te_mask])) > 1:
            aucs.append(roc_auc_score(yw[te_mask], p))
    out[str(n_rows)] = dict(
        auc_mean=float(np.mean(aucs)), auc_std=float(np.std(aucs)),
        labeled_rows_mean=float(np.mean(ns)),
        reps=[round(float(a), 3) for a in aucs])
    print(f'rows~{np.mean(ns):5.0f}  AUROC={np.mean(aucs):.3f}'
          f' ± {np.std(aucs):.3f}', flush=True)

json.dump(out, open(os.path.join(CACHE, 'e91_cross_recalib_grouped.json'),
                    'w'), indent=1)
print('-> e91_cross_recalib_grouped.json')
