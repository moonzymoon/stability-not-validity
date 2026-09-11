# -*- coding: utf-8 -*-
"""探针3: coverage-vs-M 池化口径 (与论文 47/350 池化惯例一致)."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.scm import make_sample, build_windows
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from evaluation import metrics as M
from scripts.e1_anchor import perturb_graph, stable_seed

pool = {M: [0, 0] for M in (8, 16, 32, 64)}   # [n_cert, n_cert_wrong]
for seed in range(5):
    s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                    seed=seed, noise_std=0.5, magnitude=0.5)
    w = build_windows(s)
    A = s['adjacency'].astype(float)
    scorer = make_scorer('iforest').fit(w['X_pool'])
    ctx0 = Context(w['X_pool'], scorer=scorer)
    phi0 = GRAPH_DEPENDENT['DPTA-G'](w['X_anom'], ctx0.with_graph(A))
    hits = M.hit_at_k(phi0, w['R_anom'], 3)
    etas = []
    for mi in range(64):
        rng = np.random.default_rng(stable_seed('e79', seed, 'DPTA-G', mi))
        A_m = perturb_graph(A, ['del', 'add', 'rew'][mi % 3], 0.2, rng)
        pm = GRAPH_DEPENDENT['DPTA-G'](w['X_anom'], ctx0.with_graph(A_m))
        etas.append(np.abs(pm - phi0).max(1))
    etas = np.stack(etas).T
    order = np.argsort(-phi0, axis=1)
    gap = (phi0[np.arange(len(phi0)), order[:, 2]]
           - phi0[np.arange(len(phi0)), order[:, 3]])
    for Mm in (8, 16, 32, 64):
        bound = etas[:, :Mm].max(1)
        cert = gap > 2 * bound
        pool[Mm][0] += cert.sum()
        pool[Mm][1] += (hits[cert] == 0).sum()
for Mm, (nc, nw) in pool.items():
    print(f'M={Mm}: cert {nc}/350, 池化证内错误 {nw/nc*100:.1f}%')
