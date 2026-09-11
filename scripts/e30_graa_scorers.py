# -*- coding: utf-8 -*-
"""E30: GRAA × 打分器轴 (iforest / pca / ocsvm).

动机: 基准覆盖 3 打分器, GRAA 评估只用过 iforest.
设计: SCM deviational 5 seeds × 3 scorers, true + pcmci_anom,
GRAA(p=0.4) vs PropRank/DPTA-G/zDev. 输出: _cache/e30_graa_scorers.json
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
from attribution.graph_free import GRAPH_FREE
from attribution.graa_v4 import graa_v4_attribute
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')


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

        for sk in ('iforest', 'pca', 'ocsvm'):
            scorer = make_scorer(sk).fit(w['X_pool'])
            ctx0 = Context(w['X_pool'], scorer=scorer)
            for gname, A in (('true', A_true), ('pcmci_anom', A_anom)):
                ctx_g = ctx0.with_graph(A)
                phi, info = graa_v4_attribute(
                    w['X_anom'], ctx_g, edge_conf=val_c, prune_frac=0.4)
                records.append(dict(
                    dataset=f'scm{seed}', scorer=sk, method='GRAA(p=0.4)',
                    graph_source=gname,
                    hit3=float(M.mean_hit(phi, w['R_anom'], 3))))
                for mname in ('PropRank', 'DPTA-G'):
                    phi_b = GRAPH_DEPENDENT[mname](w['X_anom'], ctx_g)
                    records.append(dict(
                        dataset=f'scm{seed}', scorer=sk, method=mname,
                        graph_source=gname,
                        hit3=float(M.mean_hit(phi_b, w['R_anom'], 3))))
            phi_z = GRAPH_FREE['zDev'](w['X_anom'], ctx0)
            records.append(dict(
                dataset=f'scm{seed}', scorer=sk, method='zDev',
                graph_source='none',
                hit3=float(M.mean_hit(phi_z, w['R_anom'], 3))))
            print(f'scm{seed} {sk} done', flush=True)

    json.dump(records, open(os.path.join(CACHE, 'e30_graa_scorers.json'), 'w'),
              default=float)
    agg = collections.defaultdict(list)
    for r in records:
        agg[(r['scorer'], r['method'], r['graph_source'])].append(r['hit3'])
    print('\n=== GRAA x scorer (hit@3, 5 seeds) ===')
    for k in sorted(agg, key=str):
        print(f'  {str(k):46s} {np.mean(agg[k]):.3f}')


if __name__ == '__main__':
    run()
