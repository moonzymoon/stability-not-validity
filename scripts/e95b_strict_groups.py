# -*- coding: utf-8 -*-
"""E95b: 严格分组下的特征组消融 (R7 任务书 2j, 缓存重分析, 非新实验).

e95 已给严格 leave-one-trajectory-out 全特征 AUROC 0.588; e77 已给
部署态(跨配置)特征组消融. 本脚本补中间格: 严格分组协议下逐组
drop / only-consistency, 检验 "一致性跨轨迹失效但形状保留微弱信号" 假设.
特征分组与 e77 相同: graph=g_, consistency=c_, attribution=a_,
scorer=s_, data=d_. 数据: _cache/e3_windows.json (与 e95 同源同折).
-> _cache/e95b_strict_groups.json
"""
import json
import os
import re

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '..', '_cache')

GROUPS = {'graph': 'g_', 'consistency': 'c_', 'attribution': 'a_',
          'scorer': 's_', 'data': 'd_'}

rows = json.load(open(os.path.join(CACHE, 'e3_windows.json'),
                      encoding='utf-8'))
allf = [c for c in rows[0] if len(c) > 2 and c[0] in 'gcasd'
        and c[1] == '_']
y = np.array([r['label'] for r in rows])
dcol = np.array([r['dataset'] for r in rows])


def group_of(d):
    if d.startswith('scm'):
        return 'scm' + re.search(r'\d+', d).group(0)
    return 'te_' + d.split('_')[0].replace('te', '')


g = np.array([group_of(d) for d in dcol])
folds = sorted(set(g))


def strict_auroc(keep_mask):
    """严格 LODO AUROC, 只用 keep_mask 为真的特征列."""
    X = np.array([[r.get(f, np.nan) for f in allf if keep_mask(f)]
                  for r in rows])
    oof = np.full(len(y), np.nan)
    for fv in folds:
        tr, te = g != fv, g == fv
        clf = HistGradientBoostingClassifier(random_state=0)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    m = ~np.isnan(oof)
    return float(roc_auc_score(y[m], oof[m]))


out = {'protocol': 'strict leave-one-trajectory-out (same folds as e95)',
       'folds': folds}
out['full'] = strict_auroc(lambda f: True)
for name, pref in GROUPS.items():
    out[f'drop_{name}'] = strict_auroc(lambda f, p=pref: not f.startswith(p))
out['only_consistency'] = strict_auroc(lambda f: f.startswith('c_'))

for k, v in out.items():
    if isinstance(v, float):
        print(f'{k}: {v:.3f}')
json.dump(out, open(os.path.join(CACHE, 'e95b_strict_groups.json'), 'w'),
          indent=1)
print('-> e95b_strict_groups.json')
