# -*- coding: utf-8 -*-
"""探针: 定位 e30(0.570) vs e33/e46(0.629) 分歧源 — scm0 pcmci_anom."""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.scm import make_sample, build_windows
from graphs.sources import graph_sources, pcmci_graph
from scorers import make_scorer
from attribution import Context
from attribution.graa_v4 import graa_v4_attribute
from evaluation import metrics as M

s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2, seed=0,
                noise_std=0.5, magnitude=0.5)
w = build_windows(s)
r_n = pcmci_graph(s['normal'])
val = np.abs(r_n['val_matrix']).max(2)
val = val / (val.max() + 1e-9)
scorer = make_scorer('iforest').fit(w['X_pool'])
ctx0 = Context(w['X_pool'], scorer=scorer)

A_e30 = pcmci_graph(s['series'])['adj'].astype(float)
A_gs = graph_sources(s)['pcmci_anom'].astype(float)
print('adj identical:', np.array_equal(A_e30, A_gs),
      '| edges:', int(A_e30.sum()), int(A_gs.sum()))

for tag, A in (('e30-fresh', A_e30), ('graph_sources', A_gs)):
    ctx = ctx0.with_graph(A)
    phi, info = graa_v4_attribute(w['X_anom'], ctx, edge_conf=val,
                                  prune_frac=0.4)
    print(tag, 'hit3=', round(float(M.mean_hit(phi, w['R_anom'], 3)), 4),
          'w_prop=', round(info['w_prop'], 3),
          'n_pruned=', info['n_pruned'])
