# -*- coding: utf-8 -*-
"""E39: 证书覆盖率的经验刻画 (Corollary 2 落地).

对 PropRank / DPTA-G, 在 SCM 5 seeds 上按扰动强度 q ∈ {0.05,0.1,0.2,0.3}
重算每窗口 gamma (top3-top4 gap) 与 eta (扰动最大位移), 输出:
  - 覆盖率 Pr[gamma > 2*eta] 随 q 的变化曲线
  - 持证窗口中 stable-wrong 占比 (与 36% 主张一致性)
  - 边界密度: 2*eta/gamma 比值分布 (等式条件可否达成的经验证据)
输出 _cache/e39_certcov.json
"""
import os
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
QS = (0.05, 0.1, 0.2, 0.3)


def run():
    out = {'curves': {}, 'boundary': {}, 'pooled': {}}
    ratios = {'PropRank': [], 'DPTA-G': []}
    for mname in ('PropRank', 'DPTA-G'):
        fn = GRAPH_DEPENDENT[mname]
        curve = []
        for q in QS:
            cov, wrong, ntot = [], [], 0
            for seed in range(5):
                s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                                seed=seed, noise_std=0.5, magnitude=0.5)
                w = build_windows(s)
                A = s['adjacency'].astype(float)
                scorer = make_scorer('iforest').fit(w['X_pool'])
                ctx0 = Context(w['X_pool'], scorer=scorer)
                phi0 = fn(w['X_anom'], ctx0.with_graph(A))
                hits = M.hit_at_k(phi0, w['R_anom'], 3)
                phis = []
                for mi in range(8):
                    rng = np.random.default_rng(stable_seed('e39', seed,
                                                            mname, mi, q))
                    A_m = perturb_graph(A, ['del', 'add', 'rew'][mi % 3],
                                        q, rng)
                    phis.append(fn(w['X_anom'], ctx0.with_graph(A_m)))
                D = np.stack(phis) - phi0[None]
                delta = np.abs(D).max(0)              # (n, d)
                order = np.argsort(-phi0, axis=1)
                for i in range(len(phi0)):
                    gap = phi0[i, order[i, 2]] - phi0[i, order[i, 3]]
                    eta = delta[i].max()
                    certified = gap > 2 * eta
                    cov.append(certified)
                    if certified:
                        wrong.append(hits[i] == 0)
                        ratios[mname].append(2 * eta / max(gap, 1e-12))
                    ntot += 1
            curve.append(dict(q=q, coverage=float(np.mean(cov)),
                              certified_wrong=float(np.mean(wrong))
                              if wrong else None, n=ntot,
                              n_certified=int(np.sum(cov))))
            print(mname, curve[-1], flush=True)
        out['curves'][mname] = curve
    for k, v in ratios.items():
        v = np.array(v)
        if len(v) == 0:
            out['boundary'][k] = dict(mean=None, median=None, q90=None,
                                      frac_above_1=None, n=0)
            continue
        out['boundary'][k] = dict(
            mean=float(v.mean()), median=float(np.median(v)),
            q90=float(np.quantile(v, 0.9)),
            frac_above_1=float((v >= 1.0).mean()), n=int(len(v)))
    json.dump(out, open(os.path.join(CACHE, 'e39_certcov.json'), 'w'))
    print('-> e39_certcov.json')


if __name__ == '__main__':
    run()
