# -*- coding: utf-8 -*-
"""E27: knee 附近细化 edge-deletion 扫描.

现有网格 del ∈ {0.1, 0.2, 0.3, 0.5} 点太稀, knee(≈30%) 定位粗糙.
补 {0.05, 0.15, 0.25, 0.35, 0.45} × {PropRank, DPTA-G} × {true, pcmci_anom}
× 5 SCM seeds × 8 次扰动 → 合并后 0.05–0.5 共 9 点细化曲线.
输出: _cache/e27_knee_fine.json
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
from graphs.sources import pcmci_graph
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from evaluation import metrics as M
from scripts.e1_anchor import perturb_graph, stable_seed

CACHE = os.path.join(SRC, '_cache')
FINE = [0.05, 0.15, 0.25, 0.35, 0.45]
M_PERTURB = 8


def run():
    records = []
    for seed in range(5):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                        seed=seed, noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        A_true = s['adjacency'].astype(float)
        A_anom = pcmci_graph(s['series'])['adj']
        scorer = make_scorer('iforest').fit(w['X_pool'])
        ctx0 = Context(w['X_pool'], scorer=scorer)

        for gname, A_base in (('true', A_true), ('pcmci_anom', A_anom)):
            for mname in ('PropRank', 'DPTA-G', 'GraphGranger'):
                fn = GRAPH_DEPENDENT[mname]
                phi_b = fn(w['X_anom'], ctx0.with_graph(A_base))
                records.append(dict(
                    dataset=f'scm{seed}', method=mname, graph_source=gname,
                    strength=0.0, family='del',
                    hit3=float(M.mean_hit(phi_b, w['R_anom'], 3)),
                    acr3=1.0))
                for st in FINE:
                    acrs, hps = [], []
                    for mi in range(M_PERTURB):
                        rng = np.random.default_rng(stable_seed(
                            'e27', seed, mname, gname, st, mi))
                        A_m = perturb_graph(A_base, 'del', st, rng)
                        phi_m = fn(w['X_anom'], ctx0.with_graph(A_m))
                        acrs.append(M.acr_at_k(phi_m, phi_b, 3).mean())
                        hps.append(float(M.mean_hit(phi_m, w['R_anom'], 3)))
                    records.append(dict(
                        dataset=f'scm{seed}', method=mname, graph_source=gname,
                        strength=st, family='del',
                        hit3=float(np.mean(hps)), acr3=float(np.mean(acrs))))
                print(f'scm{seed} {mname} {gname} fine sweep done', flush=True)

    json.dump(records, open(os.path.join(CACHE, 'e27_knee_fine.json'), 'w'),
              default=float)
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in records:
        agg[(r['method'], r['graph_source'])][r['strength']].append(
            (r['acr3'], r['hit3']))
    print('\n=== fine deletion sweep (5 seeds, mean ACR@3 / perturbed hit@3) ===')
    for k in sorted(agg, key=str):
        line = ' '.join(f'{s:.2f}:{np.mean([a for a, _ in v]):.2f}/'
                        f'{np.mean([h for _, h in v]):.2f}'
                        for s, v in sorted(agg[k].items()))
        print(f'  {str(k):32s} {line}')


if __name__ == '__main__':
    run()
