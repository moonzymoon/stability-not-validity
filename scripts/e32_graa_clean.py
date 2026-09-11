# -*- coding: utf-8 -*-
"""E32: GRAA(p=0.4) 在 pcmci_clean 图上 (补 Table 1 缺行) — 5 seeds, iforest."""
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
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graa_v4 import graa_v4_attribute
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')

per_seed = {}
win_hits = {}
for seed in range(5):
    s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                    seed=seed, noise_std=0.5, magnitude=0.5)
    w = build_windows(s)
    A_clean = pcmci_graph(s['normal'])['adj']
    val_c = np.abs(pcmci_graph(s['normal'])['val_matrix']).max(2)
    val_c = val_c / (val_c.max() + 1e-9)
    scorer = make_scorer('iforest').fit(w['X_pool'])
    ctx_g = Context(w['X_pool'], scorer=scorer).with_graph(A_clean)
    phi, info = graa_v4_attribute(w['X_anom'], ctx_g, edge_conf=val_c,
                                  prune_frac=0.4, gamma=1.0)
    h3 = M.hit_at_k(phi, w['R_anom'], 3).astype(float)
    h1 = M.hit_at_k(phi, w['R_anom'], 1).astype(float)
    per_seed[seed] = dict(hit1=float(h1.mean()), hit3=float(h3.mean()))
    win_hits[seed] = h3.tolist()
    print(f'scm{seed} clean: GRAA hit@3={h3.mean():.3f} w={info["w_prop"]:.2f}',
          flush=True)

# seed 级 bootstrap CI (B=2000)
rng = np.random.default_rng(0)
vals = np.array([per_seed[s]['hit3'] for s in range(5)])
bs = [vals[rng.integers(0, 5, 5)].mean() for _ in range(2000)]
lo, hi = np.percentile(bs, [2.5, 97.5])
out = dict(per_seed=per_seed, win_hits=win_hits,
           hit3_mean=float(vals.mean()),
           hit1_mean=float(np.mean([per_seed[s]['hit1'] for s in range(5)])),
           ci_lo=float(lo), ci_hi=float(hi))
json.dump(out, open(os.path.join(CACHE, 'e32_graa_clean.json'), 'w'), indent=1)
print(f'GRAA clean: hit@3={vals.mean():.3f} CI [{lo:.3f},{hi:.3f}]')
