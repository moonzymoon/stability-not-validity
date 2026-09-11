# -*- coding: utf-8 -*-
"""E22: GRAA v4 (confidence pruning) 快速评估."""
import os, sys, json, collections
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

        scorer = make_scorer('iforest').fit(w['X_pool'])
        ctx0 = Context(w['X_pool'], scorer=scorer)

        for gname, A in (('true', A_true), ('pcmci_anom', A_anom)):
            ctx_g = ctx0.with_graph(A)

            # GRAA v4 参数扫描
            for pf in (0.2, 0.4, 0.5):
                phi, info = graa_v4_attribute(w['X_anom'], ctx_g,
                                               edge_conf=val_c,
                                               prune_frac=pf, gamma=1.0)
                h3 = M.mean_hit(phi, w['R_anom'], 3)
                records.append(dict(
                    dataset=f'scm{seed}', method=f'GRAA-v4(p={pf})',
                    graph_source=gname, hit3=h3,
                    n_pruned=info['n_pruned'], w_prop=info['w_prop']))
                print(f'  scm{seed} GRAA-v4(p={pf}) {gname}: '
                      f'hit@3={h3:.3f} pruned={info["n_pruned"]} '
                      f'w={info["w_prop"]:.2f}', flush=True)

            # 基线
            for mname in ('PropRank', 'DPTA-G'):
                phi_b = GRAPH_DEPENDENT[mname](w['X_anom'], ctx_g)
                records.append(dict(
                    dataset=f'scm{seed}', method=mname,
                    graph_source=gname,
                    hit3=M.mean_hit(phi_b, w['R_anom'], 3)))
        phi_z = GRAPH_FREE['zDev'](w['X_anom'], ctx0)
        records.append(dict(dataset=f'scm{seed}', method='zDev',
                           graph_source='none',
                           hit3=M.mean_hit(phi_z, w['R_anom'], 3)))

    json.dump(records, open(os.path.join(CACHE, 'e22_graa_v4.json'), 'w'),
              default=float)
    agg = collections.defaultdict(list)
    for r in records:
        agg[(r['method'], r['graph_source'])].append(r['hit3'])
    print('\n=== GRAA v4 (SCM medium, hit@3) ===')
    for k in sorted(agg, key=str):
        print(f'  {str(k):38s} hit={np.mean(agg[k]):.3f}')


if __name__ == '__main__':
    run()
