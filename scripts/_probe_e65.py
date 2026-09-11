# -*- coding: utf-8 -*-
"""探针: AttNN 训练的非确定性来源 (同进程双训 vs 跨进程)."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.scm import make_sample, build_windows
from scorers import make_scorer
from attribution import Context
from attribution.attnn import attnn_alpha


def one(tag):
    sample = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                         seed=1, noise_std=0.5, magnitude=0.5)
    win = build_windows(sample, window=16, stride=20)
    scorer = make_scorer('iforest').fit(win['X_pool'])
    ctx = Context(win['X_pool'], scorer=scorer)
    alpha_fn, info = attnn_alpha(win['X_anom'], ctx, seed=20260906)
    phi = alpha_fn(win['X_anom'])
    h = hash(phi.tobytes())
    print(f'{tag}: info={json.dumps(info, default=float)[:90]} '
          f'phi_hash={h % 10**12} phi[0,:3]={np.round(phi[0,:3],6)}')
    return h % 10**12


if len(sys.argv) > 1 and sys.argv[1] == 'twice':
    one('inproc-1')
    one('inproc-2')
else:
    one('proc')
