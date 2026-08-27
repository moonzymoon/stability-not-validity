# -*- coding: utf-8 -*-
"""M 收敛性快检: 同一配置 M=8 vs M=32 的 ACR@3 差异."""
import os
import sys

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
from scripts.e1_anchor import perturb_graph

s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                seed=0, noise_std=0.5, magnitude=0.5)
w = build_windows(s)
A_true = s['adjacency'].astype(float)
scorer = make_scorer('iforest').fit(w['X_pool'])
ctx = Context(w['X_pool'], scorer=scorer)

for mname in ('PropRank', 'DPTA-G'):
    fn = GRAPH_DEPENDENT[mname]
    phi_b = fn(w['X_anom'], ctx.with_graph(A_true))
    for fam, st in (('del', 0.3), ('add', 0.5), ('rew', 0.5)):
        acrs = []
        for mi in range(32):
            rng = np.random.default_rng(9000 + mi)
            A_m = perturb_graph(A_true, fam, st, rng)
            phi_m = fn(w['X_anom'], ctx.with_graph(A_m))
            acrs.append(M.acr_at_k(phi_m, phi_b, 3).mean())
        acrs = np.array(acrs)
        m8 = [acrs[i*4:(i+1)*4].mean() for i in range(4)]  # 4 组独立 M=8
        print(f'{mname} {fam}@{st}: M=32 ACR={acrs.mean():.4f}; '
              f'独立 M=8 组: {[round(x,4) for x in m8]}; '
              f'最大组间差={max(m8)-min(m8):.4f}')
