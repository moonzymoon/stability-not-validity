# -*- coding: utf-8 -*-
"""E80: DPTA-G 零分截断 vs 伪稳定 (元宝原④, 未回应项).

检验 "伪稳定部分源于 clamp 精确零长尾" 假设:
SCM medium 5 seeds, true 图, q=0.2, DPTA-G/PropRank:
  (a) phi 精确零/近零(<1e-6/<1e-3)比例
  (b) 窗口级零分比例与窗口 ACR 的 Spearman 相关
  (c) 若零比例≈0 → 假设被驳回 (伪稳定是机制性的, 不是数值假象)
输出: _cache/e80_zero_clamp.json
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
import sys
import json
from scipy.stats import spearmanr

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.scm import make_sample, build_windows
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from scripts.e1_anchor import perturb_graph, stable_seed

CACHE = os.path.join(SRC, '_cache')


def run():
    out = []
    for seed in range(5):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                        seed=seed, noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        A = s['adjacency'].astype(float)
        scorer = make_scorer('iforest').fit(w['X_pool'])
        ctx0 = Context(w['X_pool'], scorer=scorer)
        for mname in ('DPTA-G', 'PropRank'):
            fn = GRAPH_DEPENDENT[mname]
            phi0 = fn(w['X_anom'], ctx0.with_graph(A))
            phis = []
            for mi in range(8):
                rng = np.random.default_rng(stable_seed(
                    'e80', seed, mname, mi))
                A_m = perturb_graph(A, ['del', 'add', 'rew'][mi % 3], 0.2, rng)
                phis.append(fn(w['X_anom'], ctx0.with_graph(A_m)))
            zf_exact = float(np.mean(phi0 == 0))
            zf_1e6 = float(np.mean(np.abs(phi0) < 1e-6))
            zf_1e3 = float(np.mean(np.abs(phi0) < 1e-3))
            # 窗口级: 零分比例 vs 窗口 ACR
            wz = (np.abs(phi0) < 1e-3).mean(1)
            wacr = []
            for i in range(len(phi0)):
                t0 = set(np.argsort(-phi0[i])[:3])
                ovs = [len(set(np.argsort(-pm[i])[:3]) & t0) / 3
                       for pm in phis]
                wacr.append(np.mean(ovs))
            rho, p = spearmanr(wz, wacr)
            out.append(dict(seed=seed, method=mname,
                            zero_exact=zf_exact, zero_1e6=zf_1e6,
                            zero_1e3=zf_1e3,
                            spearman_zerofrac_acr=[round(float(rho), 3),
                                                   round(float(p), 4)]))
            print(f'seed{seed} {mname}: exact={zf_exact:.4f} '
                  f'<1e-6={zf_1e6:.4f} <1e-3={zf_1e3:.4f} '
                  f'rho(zf,ACR)={rho:+.3f} p={p:.3f}')
    json.dump(out, open(os.path.join(CACHE, 'e80_zero_clamp.json'), 'w'),
              default=float, indent=1)
    print('-> e80_zero_clamp.json')


if __name__ == '__main__':
    run()
