# -*- coding: utf-8 -*-
"""E21: GRAA v3 (graph repair) 评估 — 与所有基线比较."""
import os, sys, json, collections, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.scm import make_sample, build_windows
from graphs.sources import pcmci_graph
from graphs import graph_ops as go
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graph_free import GRAPH_FREE
from attribution.graa_v3 import graa_v3_attribute
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')


def run():
    records = []
    for seed in range(3):  # 3 seeds(逐边评估计算量大)
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                        seed=seed, noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        A_true = s['adjacency'].astype(float)
        A_anom = pcmci_graph(s['series'])['adj']

        scorer = make_scorer('iforest').fit(w['X_pool'])
        ctx0 = Context(w['X_pool'], scorer=scorer)

        for gname, A in (('true', A_true), ('pcmci_anom', A_anom)):
            ctx_g = ctx0.with_graph(A)
            t0 = time.time()

            # GRAA v3
            phi, info = graa_v3_attribute(w['X_anom'], ctx_g,
                                           repair_threshold=0.001,
                                           fallback_cons=0.5)
            h3 = M.mean_hit(phi, w['R_anom'], 3)
            h1 = M.mean_hit(phi, w['R_anom'], 1)
            records.append(dict(
                dataset=f'scm{seed}', method='GRAA-v3', graph_source=gname,
                hit1=h1, hit3=h3, hit5=M.mean_hit(phi, w['R_anom'], 5),
                n_removed=info['n_edges_removed'],
                base_cons=info['base_consistency'],
                rep_cons=info['repaired_consistency'],
                fallback=info['fallback']))
            print(f'  scm{seed} GRAA-v3 {gname}: hit@3={h3:.3f} '
                  f'removed={info["n_edges_removed"]} '
                  f'cons={info["base_consistency"]:.2f}→{info["repaired_consistency"]:.2f} '
                  f'fb={info["fallback"]} ({time.time()-t0:.0f}s)', flush=True)

            # 基线
            for mname in ('PropRank', 'DPTA-G'):
                phi_b = GRAPH_DEPENDENT[mname](w['X_anom'], ctx_g)
                records.append(dict(
                    dataset=f'scm{seed}', method=mname, graph_source=gname,
                    hit3=M.mean_hit(phi_b, w['R_anom'], 3)))
            phi_z = GRAPH_FREE['zDev'](w['X_anom'], ctx0)
            records.append(dict(
                dataset=f'scm{seed}', method='zDev', graph_source='none',
                hit3=M.mean_hit(phi_z, w['R_anom'], 3)))

    json.dump(records, open(os.path.join(CACHE, 'e21_graa_v3.json'), 'w'),
              default=float)
    agg = collections.defaultdict(list)
    for r in records:
        agg[(r['method'], r['graph_source'])].append(r['hit3'])
    print('\n=== GRAA v3 vs baselines (SCM medium, hit@3) ===')
    for k in sorted(agg, key=str):
        print(f'  {str(k):38s} hit={np.mean(agg[k]):.3f}')
    return records


if __name__ == '__main__':
    run()
