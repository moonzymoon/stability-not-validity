# -*- coding: utf-8 -*-
"""E79: 证书 coverage-vs-M 曲线 (元宝要求1c).

SCM medium 5 seeds, true 图, q=0.2 (headline 强度), DPTA-G/PropRank.
96 个扰动实例: 前 M∈{8,16,32,64} 校准包络 (worst=max, emp50=median),
后 32 个 fresh 评估越界率/覆盖率/证内错误. 回应 "64-67%越界" 的
"要多少实例才够"问题. 输出: _cache/e79_cov_vs_m.json
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
import sys
import json
import time

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
N_INST = 96
N_FRESH = 32
MS = (8, 16, 32, 64)


def run():
    t0 = time.time()
    results = []
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
            hits = np.array([len(M2 := set(np.argsort(-r)[:3]) & roots) / 3
                             for r, roots in zip(phi0, w['R_anom'])])
            etas = []
            for mi in range(N_INST):
                rng = np.random.default_rng(stable_seed(
                    'e79', seed, mname, mi))
                A_m = perturb_graph(A, ['del', 'add', 'rew'][mi % 3], 0.2, rng)
                pm = fn(w['X_anom'], ctx0.with_graph(A_m))
                etas.append(np.abs(pm - phi0).max(1))       # (n,) 每窗
            etas = np.stack(etas).T                          # (n, 96)
            order = np.argsort(-phi0, axis=1)
            gap = (phi0[np.arange(len(phi0)), order[:, 2]]
                   - phi0[np.arange(len(phi0)), order[:, 3]])
            for M in MS:
                cal, fresh = etas[:, :M], etas[:, M_INST_EVAL:]
                wmax, wmed = cal.max(1), np.median(cal[:, ::2], 1)
                for rule, bound in (('worst', wmax), ('emp50', wmed)):
                    exceed = float(np.mean(
                        fresh.max(1) > bound * (2 if rule == 'worst' else 2)))
                    cert = gap > 2 * bound
                    cw = float((hits[cert] == 0).mean()) if cert.any() else None
                    results.append(dict(
                        seed=seed, method=mname, M=M, rule=rule,
                        fresh_exceed=round(exceed, 3),
                        coverage=round(float(cert.mean()), 3),
                        cert_wrong=round(float(cw), 3) if cw is not None
                        else None))
        print(f'seed {seed} done ({time.time()-t0:.0f}s)', flush=True)
    json.dump(results, open(os.path.join(CACHE, 'e79_cov_vs_m.json'), 'w'),
              default=float, indent=1)
    import collections
    agg = collections.defaultdict(list)
    for r in results:
        agg[(r['method'], r['M'], r['rule'])].append(
            (r['fresh_exceed'], r['coverage'], r['cert_wrong']))
    print('\n=== coverage-vs-M (mean over 5 seeds) ===')
    for k in sorted(agg, key=str):
        v = agg[k]
        cw = np.mean([x[2] for x in v if x[2] is not None])
        print(f'  {str(k):28} exceed={np.mean([x[0] for x in v]):.3f} '
              f'cov={np.mean([x[1] for x in v]):.3f} '
              f'wrong={cw if cw == cw else float("nan"):.3f}')


M_INST_EVAL = N_INST - N_FRESH     # = 64, fresh 恒为后 32

if __name__ == '__main__':
    run()
