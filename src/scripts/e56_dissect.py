# -*- coding: utf-8 -*-
"""E56: GRAA 拆解消融 — 补齐三缺失基线.

A prune-only:  剪枝图上纯 PropRank (gate=1, 无集成)
B fixed-gate:  未剪枝图 + 固定 w=0.5 集成 (无剪枝, 无自适应)
C random-prune: 随机剪同样比例边 + 自适应门控 (无置信度信号)
D full GRAA:   完整 (参照, 应复现 e33)
SCM 5 seeds x true/pcmci_anom, e33 口径. 输出: _cache/e56_dissect.json
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import sys
import json
import collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.scm import make_sample, build_windows
from graphs.sources import graph_sources, pcmci_graph
from scorers import make_scorer
from attribution import Context, normalize_rows
from attribution.graph_dep import proprank_attribute
from attribution.graph_free import zdev_attribute
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')


def graa_variant(X, ctx, edge_conf, p, mode, rng):
    A = ctx.graph
    D = X.shape[2]
    edges = [(A[i, j], edge_conf[i, j], i, j)
             for i in range(D) for j in range(D) if A[i, j] > 0]
    if mode in ('prune_only', 'full'):
        edges.sort(key=lambda x: x[1])
    else:                                   # random_prune / fixed_gate 处理
        order = rng.permutation(len(edges))
        edges = [edges[k] for k in order]
    n_prune = 0 if mode == 'fixed_gate' else int(len(edges) * p)
    A_pr = A.copy()
    for _, _, i, j in edges[:n_prune]:
        A_pr[i, j] = 0.0
    retained = [c for _, c, _, _ in (edges[n_prune:] if n_prune else edges)]
    gate = 1.0 if mode == 'prune_only' else (
        0.5 if mode == 'fixed_gate' else
        (float(np.mean(retained)) if retained else 0.0))
    phi_prop = normalize_rows(proprank_attribute(X, ctx.with_graph(A_pr)))
    nz_zdev = normalize_rows(zdev_attribute(X, ctx))
    return gate * phi_prop + (1 - gate) * nz_zdev


def run():
    records = []
    for seed in range(5):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2, seed=seed,
                        noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        src = graph_sources(s)
        r = pcmci_graph(s['normal'])
        val = np.abs(r['val_matrix']).max(2)
        val = val / (val.max() + 1e-9)
        scorer = make_scorer('iforest').fit(w['X_pool'])
        ctx0 = Context(w['X_pool'], scorer=scorer)
        for gname in ('true', 'pcmci_anom'):
            ctx = ctx0.with_graph(src[gname].astype(float))
            rng = np.random.default_rng(56)
            for mode in ('prune_only', 'fixed_gate', 'random_prune', 'full'):
                phi = graa_variant(w['X_anom'], ctx, val, 0.4, mode, rng)
                records.append(dict(seed=seed, graph=gname, mode=mode,
                                    hit3=float(M.mean_hit(phi, w['R_anom'],
                                                          3))))
        print(f'seed{seed} done', flush=True)
    json.dump(records, open(os.path.join(CACHE, 'e56_dissect.json'), 'w'),
              default=float)
    agg = collections.defaultdict(list)
    for r in records:
        agg[(r['graph'], r['mode'])].append(r['hit3'])
    print('\n=== hit@3 (5 seeds) ===')
    for k in sorted(agg, key=str):
        print(f'  {str(k):30s} {np.mean(agg[k]):.3f}')


if __name__ == '__main__':
    run()
