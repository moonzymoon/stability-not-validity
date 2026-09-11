# -*- coding: utf-8 -*-
"""E51: SNR 扫描 — 稳定错占比随观测噪声 noise_std 0.25/0.5/1.0."""
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
from evaluation import metrics as M
from scripts.e1_anchor import perturb_graph, stable_seed

CACHE = os.path.join(SRC, '_cache')
NOISES = (0.25, 0.5, 1.0)
OPS = ('del', 'add', 'rew')


def run():
    records = []
    for noise in NOISES:
        for seed in range(3):
            s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                            seed=seed, noise_std=noise, magnitude=0.5)
            w = build_windows(s)
            A = pcmci_graph(s['series'])['adj'].astype(float)
            scorer = make_scorer('iforest').fit(w['X_pool'])
            ctx0 = Context(w['X_pool'], scorer=scorer)
            ctx = ctx0.with_graph(A)
            for mname in ('PropRank', 'DPTA-G'):
                fn = GRAPH_DEPENDENT[mname]
                phi0 = fn(w['X_anom'], ctx)

                def acr_of(k):
                    rng = np.random.default_rng(
                        stable_seed('e51', noise, seed, mname, k))
                    A_m = perturb_graph(A, OPS[k % 3], 0.2, rng)
                    phi_m = fn(w['X_anom'], ctx0.with_graph(A_m))
                    return float(M.acr_at_k(phi_m, phi0, 3).mean())

                hit3 = float(M.mean_hit(phi0, w['R_anom'], 3))
                acr3 = float(np.mean([acr_of(k) for k in range(8)]))
                records.append(dict(noise=noise, seed=seed, method=mname,
                                    acr3=acr3, hit3=hit3,
                                    stable_wrong=bool(acr3 > 0.8 and
                                                      hit3 < 0.4)))
        print(f'noise={noise} done', flush=True)
    json.dump(records, open(os.path.join(CACHE, 'e51_snr.json'), 'w'),
              default=float)
    print('\n=== stable-wrong share / mean ACR / mean hit by noise ===')
    for noise in NOISES:
        rs = [r for r in records if r['noise'] == noise]
        print(f'  noise={noise}: '
              f'SW={np.mean([r["stable_wrong"] for r in rs]):.2f} '
              f'ACR={np.mean([r["acr3"] for r in rs]):.3f} '
              f'hit={np.mean([r["hit3"] for r in rs]):.3f}')


if __name__ == '__main__':
    run()
