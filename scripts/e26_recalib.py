# -*- coding: utf-8 -*-
"""E26: 预测器 few-shot 目标域再校准 (deviational→relational 边界的补救).

问题: 摘要报 zero-shot 0.54 —— 跨异常类型校准失效. 本实验量化
"多少目标域窗口可恢复判别力":
  对每个留出关系型数据集 d (LODO, 无泄漏):
    train = dev_rows (全部偏差型) + n 条从 ≠d 的关系型行中随机抽取
    test  = 关系型数据集 d
  n ∈ {0, 50, 100, 200, 400, 800}, 10 次重抽样 → mean AUROC ± sd.
输出: _cache/e26_recalib.json
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
    dev_rows = json.load(open(os.path.join(CACHE, 'e3_windows.json'),
                              encoding='utf-8'))
    rel_rows = json.load(open(os.path.join(CACHE, 'e9_relation_windows.json'),
                              encoding='utf-8'))
    feats = [c for c in dev_rows[0] if c[0] in 'gcasd' and c[1] == '_']
    rel_ds = sorted({r['dataset'] for r in rel_rows})
    rng = np.random.default_rng(0)

    NS = [0, 50, 100, 200, 400, 800]
    out = {str(n): [] for n in NS}
    for d in rel_ds:
        te_rows = [r for r in rel_rows if r['dataset'] == d]
        tr_pool = [r for r in rel_rows if r['dataset'] != d]
        Xte, yte = _xy(te_rows, feats)
        if yte.std() == 0:
            continue
        for rep in range(10):
            for n in NS:
                add = (rng.choice(len(tr_pool), size=min(n, len(tr_pool)),
                                  replace=False).tolist() if n > 0 else [])
                tr_rows = dev_rows + [tr_pool[i] for i in add]
                Xtr, ytr = _xy(tr_rows, feats)
                if ytr.std() == 0:
                    continue
                clf = HistGradientBoostingClassifier(
                    max_depth=4, max_iter=300, learning_rate=0.08,
                    random_state=0)
                clf.fit(Xtr, ytr)
                p = clf.predict_proba(Xte)[:, 1]
                out[str(n)].append(float(roc_auc_score(yte, p)))
        print(f'{d} done', flush=True)

    summary = {}
    for n in NS:
        v = np.array(out[str(n)])
        summary[str(n)] = dict(auroc=float(v.mean()), sd=float(v.std()),
                               n_eval=int(len(v)))
        print(f'n_target={n:4d}: AUROC={v.mean():.3f} ± {v.std():.3f} '
              f'({len(v)} evals)')
    json.dump({'summary': summary, 'raw': out},
              open(os.path.join(CACHE, 'e26_recalib.json'), 'w'), indent=1)


if __name__ == '__main__':
    run()
