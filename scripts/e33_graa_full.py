# -*- coding: utf-8 -*-
"""E33: GRAA(p=0.4) 完整指标 (hit@1/3/5 + 部署 ACR@3) — 三图源, 供 Table 1."""
import os
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
from attribution.graa_v4 import graa_v4_attribute
from evaluation import metrics as M
from scripts.e1_anchor import perturb_graph, stable_seed

CACHE = os.path.join(SRC, '_cache')
M_PERTURB = 4
FAMS = {'del': [0.1, 0.2, 0.3, 0.5], 'add': [0.2, 0.5], 'rew': [0.2, 0.5]}

per_seed = {}
for seed in range(5):
    s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                    seed=seed, noise_std=0.5, magnitude=0.5)
    w = build_windows(s)
    graphs = {'true': s['adjacency'].astype(float),
              'pcmci_clean': pcmci_graph(s['normal'])['adj'],
              'pcmci_anom': pcmci_graph(s['series'])['adj']}
    val_c = np.abs(pcmci_graph(s['normal'])['val_matrix']).max(2)
    val_c = val_c / (val_c.max() + 1e-9)
    scorer = make_scorer('iforest').fit(w['X_pool'])
    ctx0 = Context(w['X_pool'], scorer=scorer)
    per_seed[seed] = {}
    for gname, A in graphs.items():
        ctx_g = ctx0.with_graph(A)
        phi_b, _ = graa_v4_attribute(w['X_anom'], ctx_g, edge_conf=val_c,
                                     prune_frac=0.4)
        h1 = float(M.mean_hit(phi_b, w['R_anom'], 1))
        h3 = float(M.mean_hit(phi_b, w['R_anom'], 3))
        h5 = float(M.mean_hit(phi_b, w['R_anom'], 5))
        acrs = []
        for fam, sts in FAMS.items():
            for st in sts:
                for mi in range(M_PERTURB):
                    rng = np.random.default_rng(stable_seed(
                        'e33', seed, gname, fam, st, mi))
                    A_m = perturb_graph(A, fam, st, rng)
                    phi_m, _ = graa_v4_attribute(
                        w['X_anom'], ctx0.with_graph(A_m),
                        edge_conf=val_c, prune_frac=0.4)
                    acrs.append(M.acr_at_k(phi_m, phi_b, 3).mean())
        per_seed[seed][gname] = dict(
            hit1=h1, hit3=h3, hit5=h5, acr=float(np.mean(acrs)))
        print(f'scm{seed} {gname}: h1={h1:.3f} h3={h3:.3f} h5={h5:.3f} '
              f'acr={np.mean(acrs):.3f}', flush=True)

out = {'per_seed': per_seed}
for g in ('true', 'pcmci_clean', 'pcmci_anom'):
    for k in ('hit1', 'hit3', 'hit5', 'acr'):
        v = np.array([per_seed[s][g][k] for s in range(5)])
        rng = np.random.default_rng(0)
        bs = [v[rng.integers(0, 5, 5)].mean() for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[f'{g}/{k}'] = dict(mean=float(v.mean()), lo=float(lo),
                               hi=float(hi))
        print(f'{g}/{k}: {v.mean():.3f} [{lo:.3f},{hi:.3f}]')
json.dump(out, open(os.path.join(CACHE, 'e33_graa_full.json'), 'w'), indent=1)
