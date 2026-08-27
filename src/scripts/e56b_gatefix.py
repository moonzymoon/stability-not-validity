# -*- coding: utf-8 -*-
"""E56b: 剪枝图 + 固定门控 w=0.5 — 隔离'自适应门控 vs 自我放大'.

与 e56 相同协议; 新增模式 pruned_fixed: 置信度剪枝后固定 w=0.5.
输出: _cache/e56b.json
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
            A = src[gname].astype(float)
            ctx = ctx0.with_graph(A)
            D = A.shape[0]
            edges = [(A[i, j], val[i, j], i, j)
                     for i in range(D) for j in range(D) if A[i, j] > 0]
            edges.sort(key=lambda x: x[1])
            n_pr = int(len(edges) * 0.4)
            A_pr = A.copy()
            for _, _, i, j in edges[:n_pr]:
                A_pr[i, j] = 0.0
            retained = [c for _, c, _, _ in edges[n_pr:]]
            gate = float(np.mean(retained)) if retained else 0.0
            phi_prop = normalize_rows(proprank_attribute(
                w['X_anom'], ctx.with_graph(A_pr)))
            nz = normalize_rows(zdev_attribute(w['X_anom'], ctx))
            for tag, g in (('adaptive', gate), ('fixed0.5', 0.5)):
                phi = g * phi_prop + (1 - g) * nz
                records.append(dict(seed=seed, graph=gname, gate=tag,
                                    gate_val=round(g, 3),
                                    hit3=float(M.mean_hit(phi, w['R_anom'], 3))))
        print(f'seed{seed} done', flush=True)
    json.dump(records, open(os.path.join(CACHE, 'e56b.json'), 'w'),
              default=float)
    agg = collections.defaultdict(list)
    for r in records:
        agg[(r['graph'], r['gate'])].append(r['hit3'])
    for k in sorted(agg, key=str):
        print(k, round(float(np.mean(agg[k])), 3))


if __name__ == '__main__':
    run()
