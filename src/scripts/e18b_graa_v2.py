# -*- coding: utf-8 -*-
"""E18b: GRAA v2评估 — PropRank base + consistency gating."""
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
from attribution.graa_v2 import graa_v2_attribute
from evaluation import metrics as M
from scripts.e1_anchor import perturb_graph, stable_seed

CACHE = os.path.join(SRC, '_cache')


def run():
    records = []
    for seed in range(5):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                        seed=seed, noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        A_true = s['adjacency'].astype(float)
        A_pcmci = pcmci_graph(s['normal'])['adj']
        A_anom = pcmci_graph(s['series'])['adj']
        val_c = np.abs(pcmci_graph(s['normal'])['val_matrix']).max(2)
        val_c = val_c / (val_c.max() + 1e-9)

        scorer = make_scorer('iforest').fit(w['X_pool'])
        ctx0 = Context(w['X_pool'], scorer=scorer)

        for gname, A in (('true', A_true), ('pcmci_anom', A_anom)):
            ctx_g = ctx0.with_graph(A)
            for gamma in (0.5, 1.0, 2.0):
                phi, info = graa_v2_attribute(w['X_anom'], ctx_g,
                                               edge_conf=val_c, gamma=gamma)
                h3 = M.mean_hit(phi, w['R_anom'], 3)
                h1 = M.mean_hit(phi, w['R_anom'], 1)
                records.append(dict(
                    dataset=f'scm{seed}', method=f'GRAA-v2(g={gamma})',
                    graph_source=gname, hit1=h1, hit3=h3,
                    w_prop=info['w_prop_mean'], cons=info['consistency_mean']))
                print(f'  scm{seed} GRAA-v2(g={gamma}) {gname}: '
                      f'hit@3={h3:.3f} w_prop={info["w_prop_mean"]:.2f}',
                      flush=True)

        # 对照: PropRank和zDev在同条件
        for gname, A in (('true', A_true), ('pcmci_anom', A_anom)):
            phi_pr = GRAPH_DEPENDENT['PropRank'](w['X_anom'], ctx0.with_graph(A))
            records.append(dict(dataset=f'scm{seed}', method='PropRank',
                               graph_source=gname,
                               hit3=M.mean_hit(phi_pr, w['R_anom'], 3)))
        phi_z = GRAPH_FREE['zDev'](w['X_anom'], ctx0)
        records.append(dict(dataset=f'scm{seed}', method='zDev',
                           graph_source='none',
                           hit3=M.mean_hit(phi_z, w['R_anom'], 3)))

    json.dump(records, open(os.path.join(CACHE, 'e18b_graa_v2.json'), 'w'),
              default=float)
    agg = collections.defaultdict(list)
    for r in records:
        agg[(r['method'], r['graph_source'])].append(r['hit3'])
    print('\n=== GRAA v2 vs baselines (SCM medium, hit@3) ===')
    for k in sorted(agg, key=str):
        print(f'  {str(k):40s} hit={np.mean(agg[k]):.3f}')


if __name__ == '__main__':
    run()
