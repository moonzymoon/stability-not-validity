# -*- coding: utf-8 -*-
"""E77: 预测器部署态特征组消融 (元宝二审附条件项).

在污染图部署态 (graph_source=pcmci_anom) 上, 逐组剔除特征重训,
回答 "0.749 靠哪组特征": 若一致性组崩而图形状/分数边际组扛住 → 是发现;
若一致性组存活 → source-swap 机制故事需修订。
输出: _cache/e77_deploy_ablation.json
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
CACHE = os.path.join(SRC, '_cache')

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

GROUPS = {'graph': 'g_', 'consistency': 'c_', 'attribution': 'a_',
          'scorer': 's_', 'data': 'd_'}


def main():
    rows = json.load(open(os.path.join(CACHE, 'e3_windows.json')))
    dep = [r for r in rows if r['graph_source'] == 'pcmci_anom']
    allf = [c for c in dep[0] if c[0] in 'gcasd' and c[1] == '_']
    X = np.array([[r.get(f, np.nan) for f in allf] for r in dep], dtype=float)
    y = np.array([r['label'] for r in dep])
    ds = np.array([r['dataset'] for r in dep])
    print(f'deployment rows: {len(dep)}')

    def lodo(feats_mask):
        aurocs = []
        for d in sorted(set(ds)):
            tr, te = ds != d, ds == d
            if y[te].std() == 0 or y[tr].std() == 0:
                continue
            clf = HistGradientBoostingClassifier(
                max_depth=4, max_iter=300, learning_rate=0.08,
                random_state=0).fit(X[tr][:, feats_mask], y[tr])
            aurocs.append(roc_auc_score(
                y[te], clf.predict_proba(X[te][:, feats_mask])[:, 1]))
        return float(np.mean(aurocs)), len(aurocs)

    out = {}
    full, n = lodo(np.ones(len(allf), bool))
    out['full'] = dict(auroc=round(full, 3), n_folds=n)
    print(f'full features:      {full:.3f} ({n} folds)')
    for g, pref in GROUPS.items():
        mask = np.array([not f.startswith(pref) for f in allf])
        a, _ = lodo(mask)
        out[f'drop_{g}'] = dict(auroc=round(a, 3))
        print(f'drop {g:12s} {a:.3f}  (Δ={a-full:+.3f})')
    # only-consistency
    mask = np.array([f.startswith('c_') for f in allf])
    a, _ = lodo(mask)
    out['only_consistency'] = dict(auroc=round(a, 3))
    print(f'only consistency:   {a:.3f}')
    json.dump(out, open(os.path.join(CACHE, 'e77_deploy_ablation.json'),
                        'w'), indent=1)
    print('-> e77_deploy_ablation.json')


if __name__ == '__main__':
    main()
