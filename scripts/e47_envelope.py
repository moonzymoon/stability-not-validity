# -*- coding: utf-8 -*-
"""E47: 证书包络的样本外有效性.

8 个训练实例估 eta_hat -> 持证(gamma>2*eta_hat); 20 个全新实例检验:
  (a) 包络违反率: 新实例位移超过 eta_hat 的窗口占比
  (b) 持证窗口 top-3 翻转率(定理预测≈0)
  (c) 未持证窗口翻转率(对照)
输出: _cache/e47_envelope.json
"""
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
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from evaluation import metrics as M
from scripts.e1_anchor import perturb_graph, stable_seed

CACHE = os.path.join(SRC, '_cache')
Q = 0.2


def run():
    out = {}
    for mname in ('PropRank', 'DPTA-G'):
        fn = GRAPH_DEPENDENT[mname]
        viol, flip_cert, flip_uncert = [], [], []
        n_cert = n_uncert = 0
        for seed in range(5):
            s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                            seed=seed, noise_std=0.5, magnitude=0.5)
            w = build_windows(s)
            A = s['adjacency'].astype(float)
            scorer = make_scorer('iforest').fit(w['X_pool'])
            ctx0 = Context(w['X_pool'], scorer=scorer)

            def phi_of(A_g):
                return fn(w['X_anom'], ctx0.with_graph(A_g))

            phi0 = phi_of(A)
            order = np.argsort(-phi0, axis=1)
            gaps = np.array([phi0[i, order[i, 2]] - phi0[i, order[i, 3]]
                             for i in range(len(phi0))])
            d_tr = np.stack([
                np.abs(phi_of(perturb_graph(
                    A, ['del', 'add', 'rew'][k % 3], Q,
                    np.random.default_rng(stable_seed('e47a', seed, mname, k)))
                ) - phi0).max(1) for k in range(8)])     # (8, n)
            eta_hat = d_tr.max(0)
            certified = gaps > 2 * eta_hat
            d_fr = np.stack([
                np.abs(phi_of(perturb_graph(
                    A, ['del', 'add', 'rew'][k % 3], Q,
                    np.random.default_rng(stable_seed('e47b', seed, mname, k)))
                ) - phi0).max(1) for k in range(20)])    # (20, n)
            viol += list((d_fr.max(0) > eta_hat))
            for i in range(len(phi0)):
                t_ref = set(order[i][:3].tolist())
                flips = any(set(np.argsort(-phi_of(perturb_graph(
                    A, ['del', 'add', 'rew'][k % 3], Q,
                    np.random.default_rng(stable_seed('e47b', seed, mname, k)))
                )[i])[:3].tolist()) != t_ref for k in range(20))
                (flip_cert if certified[i] else flip_uncert).append(flips)
            n_cert += int(certified.sum())
            n_uncert += int((~certified).sum())
        out[mname] = dict(
            envelope_violation=float(np.mean(viol)),
            certified_flip_rate=float(np.mean(flip_cert)) if flip_cert else None,
            uncertified_flip_rate=float(np.mean(flip_uncert)),
            n_certified=n_cert, n_uncertified=n_uncert)
        print(mname, out[mname], flush=True)
    json.dump(out, open(os.path.join(CACHE, 'e47_envelope.json'), 'w'))
    print('-> e47_envelope.json')


if __name__ == '__main__':
    run()
