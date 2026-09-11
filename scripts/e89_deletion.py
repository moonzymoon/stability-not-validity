# -*- coding: utf-8 -*-
"""E89: 删除式忠实性曲线 (SCM medium, #5). ROAR-lite: 删除归因 top-k 变量
(替换为池均值) 后重打分, 分数降幅曲线 + random-k 对照.
论点预测: 有效归因(高hit)删了掉分; 稳定-错误的方法删了不掉分.
-> _cache/e89_deletion.json
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.scm import make_sample, build_windows
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graph_free import GRAPH_FREE
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')
KS = (1, 2, 3, 5)


def main():
    rows = {}
    for seed in range(5):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                        seed=seed, noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        X_anom, X_pool, roots = w['X_anom'], w['X_pool'], w['R_anom']
        scorer = make_scorer('iforest').fit(X_pool)
        ctx0 = Context(X_pool, scorer=scorer)
        mu = X_pool.reshape(-1, X_pool.shape[-1]).mean(0)

        def score_of(X):
            return scorer.score(X)

        s0 = score_of(X_anom)
        for mname, fn in (list(GRAPH_DEPENDENT.items())
                          + list(GRAPH_FREE.items())):
            if mname == 'Grad':
                continue    # 需可微打分器; 删除测试用统一 iforest 重打分
            phi = (fn(X_anom, ctx0.with_graph(s['adjacency'].astype(float)))
                   if mname in GRAPH_DEPENDENT else fn(X_anom, ctx0))
            h3 = M.hit_at_k(phi, roots, 3)
            order = np.argsort(-phi, axis=1)
            for k in KS:
                drops = []
                for i in range(len(X_anom)):
                    Xd = X_anom[i:i + 1].copy()
                    for j in order[i, :k]:
                        Xd[0, :, j] = mu[j]
                    d = float((s0[i] - scorer.score(Xd)[0])
                              / max(abs(s0[i]), 1e-9))
                    drops.append(d)
                rows.setdefault(mname, {}).setdefault(k, []).extend(drops)
            rows[mname]['hit3'] = float(np.mean(h3))
            print(f'seed {seed} {mname} hit3={rows[mname]["hit3"]:.2f}',
                  flush=True)
        # random-k 对照
        rng = np.random.default_rng(seed)
        for k in (3,):
            drops = []
            for i in range(len(X_anom)):
                Xd = X_anom[i:i + 1].copy()
                for j in rng.choice(X_anom.shape[2], k, replace=False):
                    Xd[0, :, j] = mu[j]
                drops.append(float((s0[i] - scorer.score(Xd)[0])
                                   / max(abs(s0[i]), 1e-9)))
            rows.setdefault('RandomDel', {}).setdefault(k,
                                                        []).extend(drops)

    out = {}
    for m, d in rows.items():
        if m == 'RandomDel':
            v = d[3]
            out[m] = {'drop@3': float(np.mean(v)), 'n': len(v)}
        else:
            out[m] = {'hit3': d.get('hit3'),
                      **{f'drop@{k}': float(np.mean(d[k])) for k in KS
                         if k in d}}
    # drop@3 与 hit@3 的跨方法相关
    xs = [out[m]['drop@3'] for m in out if m != 'RandomDel'
          and 'drop@3' in out[m] and out[m]['hit3'] is not None]
    ys = [out[m]['hit3'] for m in out if m != 'RandomDel'
          and 'drop@3' in out[m] and out[m]['hit3'] is not None]
    if len(xs) > 2:
        out['_corr_drop3_hit3'] = float(np.corrcoef(xs, ys)[0, 1])
    json.dump(out, open(os.path.join(CACHE, 'e89_deletion.json'), 'w'),
              indent=1)
    print(json.dumps(out, indent=1)[:1200])
    print('-> e89_deletion.json')


if __name__ == '__main__':
    main()
