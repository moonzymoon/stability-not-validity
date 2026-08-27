# -*- coding: utf-8 -*-
"""E26b: target-only 训练对照 — 检验 e26 混训失败是"淹没"还是"表征不迁移".

设计: 对每个留出关系型数据集 d (LODO):
  (a) target-only: 仅用 ≠d 的 n 条关系型行训练 (无 dev 数据);
  (b) 参照: rel-only 全量 LODO (全部 ≠d 关系型行, ~3.5k).
n ∈ {50, 100, 200, 400, 800}, 10 次重抽样.
输出: _cache/e26b_targetonly.json
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

CACHE = os.path.join(SRC, '_cache')


def _xy(rows, feats):
    X = np.array([[r.get(f, np.nan) for f in feats] for r in rows])
    y = np.array([r['label'] for r in rows])
    return X, y


def run():
    rel_rows = json.load(open(os.path.join(CACHE, 'e9_relation_windows.json'),
                              encoding='utf-8'))
    feats = [c for c in rel_rows[0] if c[0] in 'gcasd' and c[1] == '_']
    rel_ds = sorted({r['dataset'] for r in rel_rows})
    rng = np.random.default_rng(1)

    NS = [50, 100, 200, 400, 800]
    out = {str(n): [] for n in NS}
    ref = []
    for d in rel_ds:
        te_rows = [r for r in rel_rows if r['dataset'] == d]
        tr_pool = [r for r in rel_rows if r['dataset'] != d]
        Xte, yte = _xy(te_rows, feats)
        if yte.std() == 0:
            continue
        # 参照: rel-only 全量
        Xtr, ytr = _xy(tr_pool, feats)
        if ytr.std() > 0:
            clf = HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                                 learning_rate=0.08,
                                                 random_state=0)
            clf.fit(Xtr, ytr)
            ref.append(float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])))
        for rep in range(10):
            for n in NS:
                idx = rng.choice(len(tr_pool), size=min(n, len(tr_pool)),
                                 replace=False)
                Xtr, ytr = _xy([tr_pool[i] for i in idx], feats)
                if ytr.std() == 0:
                    continue
                clf = HistGradientBoostingClassifier(
                    max_depth=4, max_iter=300, learning_rate=0.08,
                    random_state=0)
                clf.fit(Xtr, ytr)
                out[str(n)].append(float(roc_auc_score(
                    yte, clf.predict_proba(Xte)[:, 1])))
        print(f'{d} done', flush=True)

    summary = {'rel_only_full_LODO': dict(auroc=float(np.mean(ref)),
                                          sd=float(np.std(ref)),
                                          n_eval=len(ref))}
    print(f'rel-only full LODO (reference): {np.mean(ref):.3f} '
          f'± {np.std(ref):.3f}')
    for n in NS:
        v = np.array(out[str(n)])
        summary[str(n)] = dict(auroc=float(v.mean()), sd=float(v.std()),
                               n_eval=int(len(v)))
        print(f'target-only n={n:4d}: AUROC={v.mean():.3f} ± {v.std():.3f}')
    json.dump({'summary': summary, 'raw': out},
              open(os.path.join(CACHE, 'e26b_targetonly.json'), 'w'), indent=1)


if __name__ == '__main__':
    run()
