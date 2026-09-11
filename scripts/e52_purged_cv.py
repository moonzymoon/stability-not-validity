# -*- coding: utf-8 -*-
"""E52: 预测器净化时间CV — 训练过去→测试未来, 带净化间隔.

对每个数据集: 时间轴前70%为训练(其他数据集全部加入), 末30%为测试,
训练窗与测试窗间隔 gap=5*window (剔除相邻相关窗). 报 AUROC 对照 LODO.
输出: _cache/e52_purged.json
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import sys
import json
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

CACHE = os.path.join(SRC, '_cache')


def run():
    rows = json.load(open(os.path.join(CACHE, 'e3_windows.json'),
                          encoding='utf-8'))
    for r in rows:
        if 'idx' not in r:
            r['idx'] = rows.index(r)
    feats = [c for c in rows[0] if c[0] in 'gcasd' and c[1] == '_']
    X = np.array([[r.get(f, np.nan) for f in feats] for r in rows])
    y = np.array([r['label'] for r in rows])
    ds = np.array([r['dataset'] for r in rows])
    # 时间序近似: 每 (dataset, idx) 排序, 前70%/后30%
    order = {}
    for i, r in enumerate(rows):
        k = (r['dataset'], r.get('method', ''))
        order.setdefault(k, []).append(i)
    test_mask = np.zeros(len(rows), bool)
    for k, idxs in order.items():
        idxs = sorted(idxs, key=lambda i: rows[i]['idx'])
        cut = int(0.7 * len(idxs))
        test_mask[idxs[cut:]] = True
    # 净化: 测试窗前 gap 个同序列窗从训练剔除
    gap = 5
    drop = set()
    for k, idxs in order.items():
        idxs = sorted(idxs, key=lambda i: rows[i]['idx'])
        cut = int(0.7 * len(idxs))
        for j in idxs[cut:]:
            pos = idxs.index(j)
            drop.update(idxs[max(0, pos - gap):pos])
    train_mask = ~test_mask & np.array([i not in drop for i in range(len(rows))])
    aucs = []
    for d in sorted(set(ds[test_mask])):
        te = test_mask & (ds == d)
        tr = train_mask & (ds != d)
        if y[tr].std() == 0 or y[te].std() == 0:
            continue
        clf = HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                             learning_rate=0.08,
                                             random_state=0).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    out = dict(purged_temporal_auroc=float(np.mean(aucs)), n_test_ds=len(aucs),
               n_train=int(train_mask.sum()), n_test=int(test_mask.sum()),
               gap_windows=gap)
    print(out)
    json.dump(out, open(os.path.join(CACHE, 'e52_purged.json'), 'w'))


if __name__ == '__main__':
    run()
