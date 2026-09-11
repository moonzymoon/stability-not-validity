# -*- coding: utf-8 -*-
"""E82: E39 的十种子扩展 (证书样本扩容, 增量报告用).

协议与 e39 完全一致 (envelope rule, q in {0.05,0.1,0.2,0.3},
M=8 混合扰动), 仅轨迹种子 5->10。前 5 种子复现 e39 (校验),
后 5 为新增; 输出每种子层与合并层 -> e82_certcov10.json。
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
    out = {'per_seed': {}, 'pooled': {}}
    for mname in ('PropRank', 'DPTA-G'):
        fn = GRAPH_DEPENDENT[mname]
        for q in QS:
            for seed in range(10):
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
                delta = np.abs(D).max(0)
                order = np.argsort(-phi0, axis=1)
                cov = wrong = 0
                for i in range(len(phi0)):
                    gap = phi0[i, order[i, 2]] - phi0[i, order[i, 3]]
                    eta = delta[i].max()
                    if gap > 2 * eta:
                        cov += 1
                        wrong += int(hits[i] == 0)
                out['per_seed'].setdefault(mname, {}).setdefault(
                    q, {})[seed] = dict(n=len(phi0), n_certified=cov,
                                        n_wrong=wrong)
            # pooled for this q
            rows = out['per_seed'][mname][q]
            ncert = sum(r['n_certified'] for r in rows.values())
            nwrong = sum(r['n_wrong'] for r in rows.values())
            ntot = sum(r['n'] for r in rows.values())
            out['pooled'].setdefault(mname, []).append(dict(
                q=q, n=ntot, n_certified=ncert,
                certified_wrong=(nwrong / ncert) if ncert else None))
            print(mname, out['pooled'][mname][-1], flush=True)
    json.dump(out, open(os.path.join(CACHE, 'e82_certcov10.json'), 'w'))
    print('-> e82_certcov10.json')


if __name__ == '__main__':
    run()
