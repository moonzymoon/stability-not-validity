# -*- coding: utf-8 -*-
"""E28: GRAA v4 超参敏感性 — p × gamma 网格 (SCM, 5 seeds).

p (prune_frac) ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
gamma (混合非线性) ∈ {0.5, 1.0, 2.0}
图源: true + pcmci_anom. p=0 时不剪枝, gamma 退化为纯混合权重差异.
输出: _cache/e28_graa_sens.json
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
from attribution.graa_v4 import graa_v4_attribute
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')
PS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
GAMMAS = [0.5, 1.0, 2.0]


def run():
    records = []
    for seed in range(5):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                        seed=seed, noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        A_true = s['adjacency'].astype(float)
        A_anom = pcmci_graph(s['series'])['adj']
        val_c = np.abs(pcmci_graph(s['normal'])['val_matrix']).max(2)
        val_c = val_c / (val_c.max() + 1e-9)
        scorer = make_scorer('iforest').fit(w['X_pool'])
        ctx0 = Context(w['X_pool'], scorer=scorer)

        for gname, A in (('true', A_true), ('pcmci_anom', A_anom)):
            ctx_g = ctx0.with_graph(A)
            for p in PS:
                for g in GAMMAS:
                    phi, info = graa_v4_attribute(
                        w['X_anom'], ctx_g, edge_conf=val_c,
                        prune_frac=p, gamma=g)
                    records.append(dict(
                        dataset=f'scm{seed}', graph_source=gname,
                        p=p, gamma=g,
                        hit3=float(M.mean_hit(phi, w['R_anom'], 3)),
                        w_prop=info['w_prop']))
        print(f'scm{seed} done ({len(records)} rows)', flush=True)

    json.dump(records, open(os.path.join(CACHE, 'e28_graa_sens.json'), 'w'),
              default=float)
    agg = collections.defaultdict(list)
    for r in records:
        agg[(r['graph_source'], r['p'], r['gamma'])].append(r['hit3'])
    print('\n=== GRAA sensitivity: hit@3 mean over 5 seeds ===')
    print('graph      p     ' + '  '.join(f'g={g:<4}' for g in GAMMAS))
    for gsrc in ('true', 'pcmci_anom'):
        for p in PS:
            row = '  '.join(f'{np.mean(agg[(gsrc, p, g)]):.3f}'
                            for g in GAMMAS)
            print(f'{gsrc:10s} {p:.1f}   {row}')


if __name__ == '__main__':
    run()
