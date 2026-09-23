# -*- coding: utf-8 -*-
"""E111: WADI 200 重校准的窗口分组(无泄漏)重算.

预提交审计发现 E109/E110 的 WADI 重校准按 method x window 记录随机切分,
同一物理窗口(共272个)会同时出现在标注侧与评估侧. 本脚本按物理窗口分组重做:
标注集按整窗口分配直至 >=200 条标注记录, 评估只在完全未见的窗口上运行.
训练底座(SCM e3)与模型超参与 E110 逐字一致.
同时输出记录级切分的窗口重叠率诊断.
输出: _cache/e111_wadi_grouped.json
"""
import io
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import hypergeom

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
CACHE = os.path.join(SRC, '_cache')
PRED_PREFIX = 'gcasd'


def feats_labels(rows):
    feats = [c for c in rows[0]
             if len(c) > 2 and c[0] in PRED_PREFIX and c[1] == '_']
    X = np.array([[r.get(f, np.nan) for f in feats] for r in rows])
    y = np.array([r['label'] for r in rows])
    return X, y, feats


def hypergeom_tail(n_pop, k_good, n_draw, k_obs):
    return float(hypergeom.sf(k_obs - 1, n_pop, k_good, n_draw))


e3 = json.load(open(os.path.join(CACHE, 'e3_windows.json'),
                    encoding='utf-8'))
Xtr, ytr, feats = feats_labels(e3)
wd = json.load(open(os.path.join(CACHE, 'e85_wadi_rows.json'),
                    encoding='utf-8'))
Xwd, ywd, _ = feats_labels(wd)
wins = np.array([r['window'] for r in wd])
uniq = np.unique(wins)
print('rows', len(wd), 'windows', len(uniq))

# 诊断: E110 原记录级切分 (seed=19) 的窗口重叠率
rng0 = np.random.default_rng(19)
overlap = []
for s in range(5):
    idx = rng0.permutation(len(wd))
    lab, rest = idx[:200], idx[200:]
    lab_w = set(wins[lab])
    overlap.append(round(float(np.mean([w in lab_w for w in wins[rest]])), 4))
diag = {'record_split_window_overlap_reps': overlap,
        'record_split_window_overlap_mean':
            round(float(np.mean(overlap)), 4)}

# 分组(无泄漏)重算: 5 seeded window-group splits
res = []
for seed in range(5):
    rng = np.random.default_rng(seed)
    worder = rng.permutation(uniq)
    lab_rows, lab_wins = 0, []
    for w in worder:
        if lab_rows >= 200:
            break
        lab_wins.append(w)
        lab_rows += int((wins == w).sum())
    lab_set = set(lab_wins)
    m_lab = np.array([w in lab_set for w in wins])
    Xr = np.vstack([Xtr, Xwd[m_lab]])
    yr = np.concatenate([ytr, ywd[m_lab]])
    c = HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                       learning_rate=0.08,
                                       random_state=0)
    c.fit(Xr, yr)
    pe = c.predict_proba(Xwd[~m_lab])[:, 1]
    ye = ywd[~m_lab]
    we = wins[~m_lab]
    o = np.argsort(-pe, kind='stable')
    k3 = int(round(0.3 * len(ye)))
    t30 = float(ye[o[:k3]].mean())
    res.append({
        'labeled_rows': int(m_lab.sum()),
        'labeled_windows': len(lab_wins),
        'eval_windows': int(len(set(we.tolist()))),
        'eval_rows': int((~m_lab).sum()),
        'base': round(float(ye.mean()), 4),
        'auroc': round(float(roc_auc_score(ye, pe)), 4),
        'prauc': round(float(average_precision_score(ye, pe)), 4),
        'top30': round(t30, 4),
        'top30_p': hypergeom_tail(
            len(ye), int(round(ye.mean() * len(ye))), k3,
            int(round(t30 * k3))),
    })

out = {'diagnosis': diag,
       'grouped_recalib200': {
           'reps': res,
           'auroc_mean': round(float(np.mean([r['auroc'] for r in res])), 4),
           'auroc_min': round(float(np.min([r['auroc'] for r in res])), 4),
           'prauc_mean': round(float(np.mean([r['prauc'] for r in res])), 4),
           'base_mean': round(float(np.mean([r['base'] for r in res])), 4),
           'top30_mean': round(float(np.mean([r['top30'] for r in res])), 4),
           'top30_min': round(float(np.min([r['top30'] for r in res])), 4),
           'top30_p_max': float(np.max([r['top30_p'] for r in res])),
           'labeled_windows_mean':
               round(float(np.mean([r['labeled_windows'] for r in res])), 1),
       }}
path = os.path.join(CACHE, 'e111_wadi_grouped.json')
json.dump(out, io.open(path, 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
print('written', path)
