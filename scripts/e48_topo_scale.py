# -*- coding: utf-8 -*-
"""E48: SCM 拓扑规模扩展 d=10/20/30 — 稳定错象限占比与 GRAA 增益随维度."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.scm import make_sample, build_windows
from graphs.sources import pcmci_graph
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graa_v4 import graa_v4_attribute
from evaluation import metrics as M
from scripts.e1_anchor import perturb_graph, stable_seed

CACHE = os.path.join(SRC, '_cache')
DS = (10, 20, 30)
OPS = ('del', 'add', 'rew')


def run():
    records = []
    for d in DS:
        for seed in range(3):
            s = make_sample(n_nodes=d, T=5000, n_single=max(3, d // 3),
                            n_joint=2, seed=seed, noise_std=0.5, magnitude=0.5)
            w = build_windows(s)
            A_true = s['adjacency'].astype(float)
            A_anom = pcmci_graph(s['series'])['adj'].astype(float)
            r = pcmci_graph(s['normal'])
            val = np.abs(r['val_matrix']).max(2)
            val = val / (val.max() + 1e-9)
            scorer = make_scorer('iforest').fit(w['X_pool'])
            ctx0 = Context(w['X_pool'], scorer=scorer)
            for gname, A in (('true', A_true), ('pcmci_anom', A_anom)):
                ctx = ctx0.with_graph(A)
                for mname in ('PropRank', 'DPTA-G'):
                    fn = GRAPH_DEPENDENT[mname]
                    phi0 = fn(w['X_anom'], ctx)

                    def acr_of(k, A=A, fn=fn, phi0=phi0, ctx0=ctx0,
                               d=d, seed=seed, mname=mname):
                        rng = np.random.default_rng(
                            stable_seed('e48', d, seed, mname, k))
                        A_m = perturb_graph(A, OPS[k % 3], 0.2, rng)
                        phi_m = fn(w['X_anom'], ctx0.with_graph(A_m))
                        return float(M.acr_at_k(phi_m, phi0, 3).mean())

                    hit3 = float(M.mean_hit(phi0, w['R_anom'], 3))
                    acr3 = float(np.mean([acr_of(k) for k in range(8)]))
                    records.append(dict(d=d, seed=seed, graph=gname,
                                        method=mname, acr3=acr3, hit3=hit3,
                                        stable_wrong=bool(acr3 > 0.8 and
                                                          hit3 < 0.4)))
                phi_g, _ = graa_v4_attribute(w['X_anom'], ctx, edge_conf=val,
                                             prune_frac=0.4)
                records.append(dict(d=d, seed=seed, graph=gname,
                                    method='GRAA(p=0.4)', acr3=None,
                                    hit3=float(M.mean_hit(phi_g, w['R_anom'],
                                                          3))))
            print(f'd={d} seed{seed} done', flush=True)
    json.dump(records, open(os.path.join(CACHE, 'e48_topo.json'), 'w'),
              default=float)
    print('\n=== summary ===')
    for d in DS:
        cells = [r for r in records if r['d'] == d
                 and r['graph'] == 'pcmci_anom'
                 and r['method'] in ('PropRank', 'DPTA-G')]
        sw = float(np.mean([r['stable_wrong'] for r in cells]))
        means = {m: float(np.mean([x['hit3'] for x in records
                                   if x['d'] == d
                                   and x['graph'] == 'pcmci_anom'
                                   and x['method'] == m]))
                 for m in ('GRAA(p=0.4)', 'PropRank', 'DPTA-G')}
        print(f'  d={d}: SWshare={sw:.2f} '
              f'GRAA={means["GRAA(p=0.4)"]:.3f} '
              f'PropRank={means["PropRank"]:.3f} '
              f'DPTA-G={means["DPTA-G"]:.3f}')


if __name__ == '__main__':
    run()
