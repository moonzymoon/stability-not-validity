# -*- coding: utf-8 -*-
"""探针: 证内错误率对扰动实例种子的敏感性 (e39种子 vs e79种子)."""
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

for tag in ('e39', 'e79'):
    tot_c = tot_w = tot_n = 0
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
        for mi in range(8):
            rng = np.random.default_rng(
                stable_seed(tag, seed, 'DPTA-G', mi, 0.2)
                if tag == 'e39' else stable_seed(tag, seed, 'DPTA-G', mi))
            A_m = perturb_graph(A, ['del', 'add', 'rew'][mi % 3], 0.2, rng)
            pm = GRAPH_DEPENDENT['DPTA-G'](w['X_anom'], ctx0.with_graph(A_m))
            etas.append(np.abs(pm - phi0).max(1))
        eta = np.stack(etas).T                     # (n, 8)
        bound = eta.max(1)
        order = np.argsort(-phi0, axis=1)
        gap = (phi0[np.arange(len(phi0)), order[:, 2]]
               - phi0[np.arange(len(phi0)), order[:, 3]])
        cert = gap > 2 * bound
        tot_c += cert.sum()
        tot_w += (hits[cert] == 0).sum()
        tot_n += len(phi0)
    print(f'{tag}种子: 覆盖 {tot_c}/{tot_n} ({tot_c/tot_n*100:.1f}%), '
          f'证内错误率 {tot_w/max(tot_c,1)*100:.1f}%')
