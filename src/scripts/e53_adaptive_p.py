# -*- coding: utf-8 -*-
"""E53: 自适应剪枝比例 p* = clip(1 - mean|val|, 0.1, 0.6) vs 固定 p=0.4.

可观测规则(无测试调参): 部署图全边平均置信度低 -> 多剪.
SCM 5 seeds x 两图源; e33 口径. 输出: _cache/e53_adaptive_p.json
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.scm import make_sample, build_windows
from graphs.sources import graph_sources, pcmci_graph
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
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
        # 可观测规则依赖部署图自身置信度: 用部署图上的边置信度均值
        from attribution.graa_v4 import graa_v4_attribute
        for gname in ('pcmci_clean', 'pcmci_anom', 'true'):
            A = src[gname].astype(float)
            ctx = ctx0.with_graph(A)
            edges_val = val[A > 0]
            p_star = float(np.clip(1 - edges_val.mean(), 0.1, 0.6))
            for tag, p in (('fixed', 0.4), ('adaptive', p_star)):
                phi, _ = graa_v4_attribute(w['X_anom'], ctx, edge_conf=val,
                                           prune_frac=p)
                records.append(dict(seed=seed, graph=gname, mode=tag, p=p,
                                    hit3=float(M.mean_hit(phi, w['R_anom'],
                                                          3))))
        print(f'seed{seed} done', flush=True)
    json.dump(records, open(os.path.join(CACHE, 'e53_adaptive_p.json'), 'w'),
              default=float)
    import collections
    agg = collections.defaultdict(list)
    for r in records:
        agg[(r['graph'], r['mode'])].append(r['hit3'])
    ps = collections.defaultdict(list)
    for r in records:
        if r['mode'] == 'adaptive':
            ps[r['graph']].append(r['p'])
    print('\n=== hit@3 (5 seeds) ===')
    for k in sorted(agg, key=str):
        print(f'  {str(k):28s} {np.mean(agg[k]):.3f}')
    print('adaptive p* by graph:', {k: round(float(np.mean(v)), 2)
                                    for k, v in ps.items()})


if __name__ == '__main__':
    run()
