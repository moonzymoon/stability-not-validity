# -*- coding: utf-8 -*-
"""E91: 跨族少样本重校准 (阶段一实验1).

问题: 预测器零样本 SCM->WADI 只有 0.57 (跨族边界).
实验: 加入 n 个目标族(WADI)标注窗重训, n in {0,50,100,200,400},
     其余 WADI 窗测试; 5 次随机划分, 报 AUROC 均值±std.
     特征: e85_wadi_rows.json (WADI) + e3_windows.json (SCM).
-> _cache/e91_cross_recalib.json
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
Xw = np.array([[r.get(f, np.nan) for f in feats] for r in wad])
yw = np.array([r['label'] for r in wad])
pos = np.where(yw == 1)[0]
neg = np.where(yw == 0)[0]
print('SCM rows', len(ys), '| WADI rows', len(yw),
      '| WADI pos', len(pos), flush=True)

out = {}
for n in (0, 50, 100, 200, 400):
    aucs = []
    for rep in range(5):
        rng = np.random.default_rng(1000 + rep)
        # 分层抽 n/2 正 n/2 负 (n=0 时纯零样本)
        if n > 0:
            k = min(n // 2, len(pos), len(neg))
            tr_idx = np.concatenate([
                rng.choice(pos, k, replace=False),
                rng.choice(neg, k, replace=False)])
        else:
            tr_idx = np.array([], dtype=int)
        te_idx = np.setdiff1d(np.arange(len(yw)), tr_idx)
        clf = HistGradientBoostingClassifier(random_state=rep)
        if n > 0:
            Xtr = np.vstack([Xs, Xw[tr_idx]])
            ytr = np.concatenate([ys, yw[tr_idx]])
        else:
            Xtr, ytr = Xs, ys
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xw[te_idx])[:, 1]
        if len(set(yw[te_idx])) > 1:
            aucs.append(roc_auc_score(yw[te_idx], p))
    out[str(n)] = dict(auc_mean=float(np.mean(aucs)),
                       auc_std=float(np.std(aucs)),
                       reps=[round(float(a), 3) for a in aucs])
    print(f'n={n:4d}  AUROC={np.mean(aucs):.3f} ± {np.std(aucs):.3f}',
          flush=True)

json.dump(out, open(os.path.join(CACHE, 'e91_cross_recalib.json'), 'w'),
          indent=1)
print('-> e91_cross_recalib.json')
