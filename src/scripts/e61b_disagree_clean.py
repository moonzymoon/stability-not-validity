# -*- coding: utf-8 -*-
"""E61b: 跨方法分歧负结果在 pcmci_clean 学图上复测(堵单配置攻击点)."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
import sys
import json
import numpy as np
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.scm import make_sample, build_windows
from graphs.sources import pcmci_graph
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graph_free import GRAPH_FREE

CACHE = os.path.join(SRC, '_cache')
METHODS = [m for m in list(GRAPH_DEPENDENT) + list(GRAPH_FREE)
           if m not in ('Random', 'Grad')]

rows = []
for seed in range(5):
    s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                    seed=seed, noise_std=0.5, magnitude=0.5)
    w = build_windows(s)
    A = pcmci_graph(s['normal'])['adj'].astype(float)
    scorer = make_scorer('iforest').fit(w['X_pool'])
    ctx0 = Context(w['X_pool'], scorer=scorer).with_graph(A)
    phis = {}
    for m in METHODS:
        phis[m] = (GRAPH_DEPENDENT[m](w['X_anom'], ctx0) if m in GRAPH_DEPENDENT
                   else GRAPH_FREE[m](w['X_anom'], ctx0))
    for i in range(len(w['X_anom'])):
        tops = {m: set(np.argsort(-phis[m][i])[:3]) for m in METHODS}
        dists = [1 - len(tops[a] & tops[b]) / 3
                 for ai, a in enumerate(METHODS) for b in METHODS[ai + 1:]]
        cnt = Counter(v for m in METHODS for v in tops[m])
        consensus = set(v for v, c in cnt.most_common(3))
        R = set(np.flatnonzero(w['R_anom'][i]).tolist())
        rows.append(dict(seed=seed, window=i,
                         disagreement=float(np.mean(dists)),
                         mean_hit=float(np.mean(
                             [1 if tops[m] & R else 0 for m in METHODS])),
                         consensus_hit=int(bool(consensus & R))))
    print(f'seed {seed} done', flush=True)

json.dump(rows, open(os.path.join(CACHE, 'e61b_disagree_clean.json'), 'w'),
          default=float)
dis = np.array([r['disagreement'] for r in rows])
hit = np.array([r['mean_hit'] for r in rows])
chit = np.array([r['consensus_hit'] for r in rows])
from scipy.stats import spearmanr
rho, p = spearmanr(dis, hit)
print(f'pcmci_clean: n={len(rows)} rho={rho:.3f} p={p:.2e} '
      f'共识{chit.mean():.3f} vs 均值{hit.mean():.3f} '
      f'(+{(chit.mean()-hit.mean())*100:.1f}pp)')
