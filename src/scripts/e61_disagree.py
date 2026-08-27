# -*- coding: utf-8 -*-
"""E61: 创新3 — 跨方法分歧作为零标签效度信号.
SCM medium 5 seeds, iforest, true 图: 7 个方法 (6 家族 + Random 不含)
在同一批窗口上出 phi; 每窗口计算平均两两 top-3 Jaccard 距离 (分歧度),
检验: 分歧高的窗口是否更可能错 (方法自身的 hit@3).
附加: 多数票共识 top-3 的 hit@3 vs 最佳单方法.
输出: _cache/e61_disagree.json
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
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
METHODS = [m for m in list(GRAPH_DEPENDENT) + list(GRAPH_FREE)
           if m not in ('Random', 'Grad')]


def run():
    rows = []
    for seed in range(5):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                        seed=seed, noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        A = s['adjacency'].astype(float)
        scorer = make_scorer('iforest').fit(w['X_pool'])
        ctx0 = Context(w['X_pool'], scorer=scorer).with_graph(A)
        phis = {}
        for m in METHODS:
            if m in GRAPH_DEPENDENT:
                phis[m] = GRAPH_DEPENDENT[m](w['X_anom'], ctx0)
            else:
                phis[m] = GRAPH_FREE[m](w['X_anom'], ctx0)
        n = len(w['X_anom'])
        for i in range(n):
            tops = {m: set(np.argsort(-phis[m][i])[:3]) for m in METHODS}
            dists = []
            for a_i, ma in enumerate(METHODS):
                for mb in METHODS[a_i + 1:]:
                    dists.append(1 - len(tops[ma] & tops[mb]) / 3)
            # 共识: 出现在 >= 一半方法 top-3 中的变量, 取 3 个
            from collections import Counter
            cnt = Counter(v for m in METHODS for v in tops[m])
            consensus = set(v for v, c in cnt.most_common(3))
            R = set(np.flatnonzero(w['R_anom'][i]).tolist())
            rows.append(dict(
                seed=seed, window=i,
                disagreement=float(np.mean(dists)),
                mean_hit=float(np.mean(
                    [1 if tops[m] & R else 0 for m in METHODS])),
                consensus_hit=int(bool(consensus & R))))
        print(f'seed {seed} done', flush=True)

    json.dump(rows, open(os.path.join(CACHE, 'e61_disagree.json'), 'w'),
              default=float)

    dis = np.array([r['disagreement'] for r in rows])
    hit = np.array([r['mean_hit'] for r in rows])
    chit = np.array([r['consensus_hit'] for r in rows])
    qs = np.quantile(dis, [0.25, 0.5, 0.75])
    print(f'n={len(rows)}; 分歧四分位的平均hit: '
          f"{[round(float(hit[dis <= qs[0]].mean()), 3)]} "
          f"{[round(float(hit[(dis > qs[0]) & (dis <= qs[1])].mean()), 3)]} "
          f"{[round(float(hit[(dis > qs[1]) & (dis <= qs[2])].mean()), 3)]} "
          f"{[round(float(hit[dis > qs[2]].mean()), 3)]}")
    print(f'共识 hit@3 = {chit.mean():.3f} vs 方法均值 hit@3 = {hit.mean():.3f}')
    # Spearman
    from scipy.stats import spearmanr
    rho, p = spearmanr(dis, hit)
    print(f'Spearman(分歧, 效度) = {rho:.3f} (p={p:.2e})')


if __name__ == '__main__':
    run()
