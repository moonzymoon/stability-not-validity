# -*- coding: utf-8 -*-
"""E45: GRAA 门控聚合消融 — mean vs median vs trimmed(20%) vs max.

内联复制 graa_v4 逻辑 (30 行) 并参数化聚合函数, 避免 numpy 模块单例污染.
SCM 5 seeds x 两图源 x 聚合; 输出 _cache/e45_gate_agg.json
"""
import os
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


def normalize_rows(phi):
    phi = np.abs(phi)
    s = phi.sum(1, keepdims=True)
    return phi / (s + 1e-12)


def graa_agg(X, ctx, edge_conf, prune_frac, agg):
    A = ctx.graph
    D = X.shape[2]
    edges = [(A[i, j], edge_conf[i, j], i, j)
             for i in range(D) for j in range(D) if A[i, j] > 0]
    edges.sort(key=lambda x: x[1])
    n_prune = int(len(edges) * prune_frac)
    A_pruned = A.copy()
    for _, _, i, j in edges[:n_prune]:
        A_pruned[i, j] = 0.0
    retained = [c for _, c, _, _ in edges[n_prune:]]
    gate = float(agg(retained)) if retained else 0.0
    phi_prop = normalize_rows(proprank_attribute(X, ctx.with_graph(A_pruned)))
    nz_zdev = normalize_rows(zdev_attribute(X, ctx))
    return gate * phi_prop + (1 - gate) * nz_zdev, gate


def val_conf_e33(series_normal):
    """与 e33/主表完全一致: 正常段 PCMCI |val| 全矩阵 / max (不加 adj 罩)."""
    r = pcmci_graph(series_normal)
    v = np.abs(r['val_matrix']).max(2)
    return v / (v.max() + 1e-9)


def _trimmed(v):
    a = np.sort(np.ravel(v))
    k = max(1, len(a) // 10)
    return float(np.mean(a[k:-k])) if len(a) > 2 * k else float(np.mean(a))


AGG = {'mean': np.mean, 'median': np.median,
       'trimmed20': _trimmed, 'max': np.max}


def run():
    records = []
    for seed in range(5):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2, seed=seed,
                        noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        src = graph_sources(s)
        scorer = make_scorer('iforest').fit(w['X_pool'])
        for gname, series in (('pcmci_clean', s['normal']),
                              ('pcmci_anom', s['series'])):
            A = src[gname].astype(float)
            val = val_conf_e33(s['normal'])   # 与 e33/主表完全同口径
            ctx = Context(w['X_pool'], scorer=scorer).with_graph(A)
            for agg_name, fn in AGG.items():
                phi, gate = graa_agg(w['X_anom'], ctx, val, 0.4, fn)
                hit3 = float(M.mean_hit(phi, w['R_anom'], 3))
                records.append(dict(seed=seed, graph=gname, agg=agg_name,
                                    hit3=hit3, gate=round(gate, 4)))
        print(f'seed{seed} done', flush=True)
    json.dump(records, open(os.path.join(CACHE, 'e45_gate_agg.json'), 'w'),
              default=float)
    agg_m = collections.defaultdict(list)
    for r in records:
        agg_m[(r['graph'], r['agg'])].append(r['hit3'])
    print('\n=== hit@3 (mean over 5 seeds) ===')
    for k in sorted(agg_m, key=str):
        print(f'  {str(k):30s} {np.mean(agg_m[k]):.3f} '
              f'(sd {np.std(agg_m[k]):.3f})')


if __name__ == '__main__':
    run()
