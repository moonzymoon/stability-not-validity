# -*- coding: utf-8 -*-
"""E70 (夜航 P4): 对抗性边选择 — SS 第五测试床 (port 自 e59).

回应 "对抗结论只在 SCM 成立" 的潜在质疑: 在 RCAEval-SS 的学图
(pcmci_clean, 部署场景) 上重做贪心对抗搜索。预算 B=2 (与论文主结果一致,
随机对照 q=0.1/0.2, M=8)。单元: 每服务 1 例 cpu 故障 rep1 (6 例)。
方法: PropRank / DPTA-G (图依赖代表)。确定性: stable_seed。
输出: _cache/e70_adversarial_ss.json
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

from data.rcaeval2 import load_units
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from scripts.e1_anchor import perturb_graph, stable_seed
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')
BUDGET = 2


def topk_sets(phi, K=3):
    return [set(np.argsort(-row)[:K]) for row in phi]


def mean_jac_dist(a, b):
    return float(np.mean([1 - len(x & y) / 3 for x, y in zip(a, b)]))


def candidates(A):
    d = A.shape[0]
    dels = [(i, j, 'del') for i in range(d) for j in range(d) if A[i, j] > 0]
    adds = [(i, j, 'add') for i in range(d) for j in range(d)
            if i != j and A[i, j] == 0]
    return dels + adds


def apply_op(A, op):
    B = A.copy()
    i, j, kind = op
    if kind == 'del':
        B[i, j] = 0
    else:
        B[i, j] = 1
    return B


def greedy(A, fn, ctx0, X, s0, budget):
    A_c = A.copy()
    dist = 0.0
    for _ in range(budget):
        best, best_d = None, -1
        for op in candidates(A_c):
            A_t = apply_op(A_c, op)
            phi_t = fn(X, ctx0.with_graph(A_t))
            dd = mean_jac_dist(topk_sets(phi_t), s0)
            if dd > best_d:
                best_d, best = dd, op
        A_c = apply_op(A_c, best)
        dist = best_d
    return A_c, dist


def run():
    units_all = [u for u in load_units('ss', reps=(1,))
                 if not np.isnan(u['series_full']).any()]
    by_svc = {}
    for u in units_all:
        if u['fault'] == 'cpu' and u['service'] not in by_svc:
            by_svc[u['service']] = u
    units = [by_svc[s] for s in sorted(by_svc)][:6]
    print(f'{len(units)} units: {[u["name"] for u in units]}', flush=True)

    records = []
    for ui, u in enumerate(units):
        t0 = time.time()
        gpath = os.path.join(CACHE, f'e66_graph_clean_{u["name"]}.npz')
        d = np.load(gpath)
        A = d['A'].astype(float)
        scorer = make_scorer('iforest').fit(u['X_pool'])
        ctx0 = Context(u['X_pool'], scorer=scorer)
        for mname in ('PropRank', 'DPTA-G'):
            fn = GRAPH_DEPENDENT[mname]
            phi0 = fn(u['X_anom'], ctx0.with_graph(A))
            s0 = topk_sets(phi0)
            rec = dict(unit=u['name'], method=mname)
            # 随机对照
            for q in (0.1, 0.2):
                ovs = []
                for mi in range(8):
                    rng = np.random.default_rng(stable_seed(
                        'e70', u['name'], mname, q, mi))
                    A_m = perturb_graph(A, ['del', 'add', 'rew'][mi % 3],
                                        q, rng)
                    phi_m = fn(u['X_anom'], ctx0.with_graph(A_m))
                    ovs.append(M.acr_at_k(phi_m, phi0, 3).mean())
                rec[f'random_q{q}'] = float(np.mean(ovs))
            # 贪心对抗 B=2
            _, dist = greedy(A, fn, ctx0, u['X_anom'], s0, BUDGET)
            rec['adversarial_B2'] = float(1 - dist)
            records.append(rec)
            print(f'[{ui+1}/{len(units)}] {u["name"]} {mname}: '
                  f'rand0.1={rec["random_q0.1"]:.3f} '
                  f'rand0.2={rec["random_q0.2"]:.3f} '
                  f'advB2={rec["adversarial_B2"]:.3f} '
                  f'({time.time()-t0:.0f}s)', flush=True)
    json.dump(records, open(os.path.join(CACHE, 'e70_adversarial_ss.json'),
                            'w'), default=float, indent=1)
    import collections
    agg = collections.defaultdict(list)
    for r in records:
        for k in ('random_q0.1', 'random_q0.2', 'adversarial_B2'):
            agg[(r['method'], k)].append(r[k])
    print('\n=== E70 adversarial SS (mean over units) ===')
    for k in sorted(agg, key=str):
        print(f'  {str(k):30} {np.mean(agg[k]):.3f}')


if __name__ == '__main__':
    run()
