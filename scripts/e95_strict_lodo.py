# -*- coding: utf-8 -*-
"""E95: 严格分组 LODO (审核指控3修复).

旧 LODO 按 dataset 字符串留出 -> scm0_m0.25 与 scm0_m0.5 (同种子同轨迹)
分居两侧, 同源暴露. 严格版: 同物理种子两种强度同时留出 (SCM 5 组 +
TE 6 单元 = 11 折). 同时给出严格 OOF 的覆盖率-效度曲线 (0.34->0.67 复核).
-> _cache/e95_strict_lodo.json
"""
import json
import os
import re

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '..', '_cache')

rows = json.load(open(os.path.join(CACHE, 'e3_windows.json'),
                      encoding='utf-8'))
feats = [c for c in rows[0] if len(c) > 2 and c[0] in 'gcasd'
         and c[1] == '_']
X = np.array([[r.get(f, np.nan) for f in feats] for r in rows])
y = np.array([r['label'] for r in rows])
dcol = np.array([r['dataset'] for r in rows])


def group_of(d):
    if d.startswith('scm'):
        return 'scm' + re.search(r'\d+', d).group(0)
    return 'te_' + d.split('_')[0].replace('te', '')


g = np.array([group_of(d) for d in dcol])
groups = sorted(set(g))
print('groups:', groups, flush=True)

oof = np.full(len(y), np.nan)
for gv in groups:
    tr, te = g != gv, g == gv
    clf = HistGradientBoostingClassifier(random_state=0)
    clf.fit(X[tr], y[tr])
    oof[te] = clf.predict_proba(X[te])[:, 1]
    print(f'fold {gv}: n={te.sum()}, AUROC='
          f'{roc_auc_score(y[te], oof[te]) if len(set(y[te])) > 1 else float("nan"):.3f}',
          flush=True)

mask = ~np.isnan(oof)
strict_auc = float(roc_auc_score(y[mask], oof[mask]))
print(f'\nSTRICT leave-one-trajectory-out AUROC: {strict_auc:.3f} '
      f'(原按 dataset 字符串 LODO = 0.79)', flush=True)

# 覆盖率-效度曲线 (严格 OOF)
order = np.argsort(-oof[mask])
ym = y[mask]
n = len(ym)
curve = {}
for cov in (1.0, 0.5, 0.3):
    k = max(1, int(cov * n))
    curve[str(cov)] = float(ym[order[:k]].mean())
print('strict OOF coverage-validity:',
      {k: round(v, 3) for k, v in curve.items()}, flush=True)

# 每折窗口级 hit 分布供诊断
out = dict(strict_lodo_auc=strict_auc,
           groups=list(groups),
           per_fold={gv: float(roc_auc_score(y[g == gv], oof[g == gv]))
                     for gv in groups if len(set(y[g == gv])) > 1},
           curve=curve, n_rows=int(mask.sum()))
json.dump(out, open(os.path.join(CACHE, 'e95_strict_lodo.json'), 'w'),
          indent=1)
print('-> e95_strict_lodo.json')
