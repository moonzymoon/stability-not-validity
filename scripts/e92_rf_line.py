# -*- coding: utf-8 -*-
"""E92: RF vs HGB 学习器全线对照 (阶段一实验2). 四条战线:
(a) SCM 跨数据集 LODO (e3_windows)
(b) SS 窗口 LODO (e67_ss_windows; unit 由 window//12 重建, 72 单元
    × 12 窗 × 10 方法 = 8640 行, 两模型同折, 对照有效)
(c) WADI 零样本 (e85 行)
(d) WADI 200 实例分组重校准 (e91 协议)
不替换论文口径(决策留给阶段二), 只产对照证据 -> e92_rf_line.json
"""
import json
import os

import numpy as np
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '..', '_cache')


def feats_of(rows):
    return [c for c in rows[0] if len(c) > 2 and c[0] in 'gcasd'
            and c[1] == '_']


def mk_hgb(seed=0):
    return HistGradientBoostingClassifier(random_state=seed)


def mk_rf(seed=0):
    return make_pipeline(SimpleImputer(strategy='median'),
                         RandomForestClassifier(n_estimators=300,
                                                random_state=seed,
                                                n_jobs=-1))


out = {}

# ---- (a) SCM 跨数据集 LODO ----
scm = json.load(open(os.path.join(CACHE, 'e3_windows.json'),
                     encoding='utf-8'))
F = feats_of(scm)
X = np.array([[r.get(f, np.nan) for f in F] for r in scm])
y = np.array([r['label'] for r in scm])
dcol = np.array([r['dataset'] for r in scm])
ds = sorted(set(dcol))
res = {}
for name, mk in (('hgb', mk_hgb), ('rf', mk_rf)):
    aucs = []
    for d in ds:
        tr, te = dcol != d, dcol == d
        m = mk()
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        if len(set(y[te])) > 1:
            aucs.append(roc_auc_score(y[te], p))
    res[name] = float(np.mean(aucs))
    print(f'(a) SCM LODO {name}={np.mean(aucs):.3f}', flush=True)
out['scm_lodo'] = res

# ---- (b) SS LODO (unit = window//12) ----
try:
    ss = json.load(open(os.path.join(CACHE, 'e67_ss_windows.json'),
                        encoding='utf-8'))
    F2 = feats_of(ss)
    Xs2 = np.array([[r.get(f, np.nan) for f in F2] for r in ss])
    ys2 = np.array([r['label'] for r in ss])
    unit = np.array([r['window'] // 12 for r in ss])
    units = sorted(set(unit))
    res = {}
    for name, mk in (('hgb', mk_hgb), ('rf', mk_rf)):
        aucs = []
        for u in units:
            tr, te = unit != u, unit == u
            if te.sum() < 5 or len(set(ys2[te])) < 2:
                continue
            m = mk()
            m.fit(Xs2[tr], ys2[tr])
            p = m.predict_proba(Xs2[te])[:, 1]
            aucs.append(roc_auc_score(ys2[te], p))
        res[name] = float(np.mean(aucs))
        print(f'(b) SS LODO {name}={np.mean(aucs):.3f} '
              f'({len(aucs)} folds)', flush=True)
    out['ss_lodo'] = res
except Exception as e:
    print('(b) skip:', repr(e)[:100], flush=True)

# ---- (c) WADI 零样本 ----
wad = json.load(open(os.path.join(CACHE, 'e85_wadi_rows.json'),
                     encoding='utf-8'))
F3 = feats_of(wad)
Xw = np.array([[r.get(f, np.nan) for f in F3] for r in wad])
yw = np.array([r['label'] for r in wad])
widx = np.array([r['window'] for r in wad])
uniq = np.unique(widx)
res = {}
for name, mk in (('hgb', mk_hgb), ('rf', mk_rf)):
    m = mk()
    m.fit(X, y)
    p = m.predict_proba(Xw)[:, 1]
    res[name] = float(roc_auc_score(yw, p))
    print(f'(c) WADI zero-shot {name}={res[name]:.3f}', flush=True)
out['wadi_zeroshot'] = res

# ---- (d) 200 实例分组重校准 ----
res = {}
for name, mk in (('hgb', mk_hgb), ('rf', mk_rf)):
    aucs = []
    for rep in range(5):
        rng = np.random.default_rng(3000 + rep)
        k = 200 // 9
        sel = rng.choice(uniq, k, replace=False)
        tr = np.isin(widx, sel)
        m = mk()
        m.fit(np.vstack([X, Xw[tr]]), np.concatenate([y, yw[tr]]))
        p = m.predict_proba(Xw[~tr])[:, 1]
        aucs.append(roc_auc_score(yw[~tr], p))
    res[name] = float(np.mean(aucs))
    print(f'(d) WADI recalib200 {name}={np.mean(aucs):.3f}', flush=True)
out['wadi_recalib200'] = res

json.dump(out, open(os.path.join(CACHE, 'e92_rf_line.json'), 'w'), indent=1)
print('-> e92_rf_line.json')
