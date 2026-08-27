# -*- coding: utf-8 -*-
"""E59: 创新1 — 对抗性扰动算子: 随机 ACR 是否高估了稳定性?
SCM medium 5 seeds, true 图, PropRank/DPTA-G.
对抗 = 贪心边搜索: 每步在全部 {删一条边, 加一条非边} 候选里选使
单元级平均 top-3 Jaccard 距离最大的一步, 预算 B ∈ {2, 4} 条边
(与随机 q=0.1/0.2 的期望边改数 ~1.7/3.4 匹配).
对照: 随机算子 ACR (同预算 M=8 实例均值).
输出: _cache/e59_adversarial.json
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
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


def run():
    out = []
    for seed in range(5):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                        seed=seed, noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        A = s['adjacency'].astype(float)
        scorer = make_scorer('iforest').fit(w['X_pool'])
        ctx0 = Context(w['X_pool'], scorer=scorer)

        for mname in ('PropRank', 'DPTA-G'):
            fn = GRAPH_DEPENDENT[mname]
            phi0 = fn(w['X_anom'], ctx0.with_graph(A))
            s0 = topk_sets(phi0)

            # 随机对照 (两个强度)
            rnd = {}
            for q in (0.1, 0.2):
                ovs = []
                for mi in range(8):
                    rng = np.random.default_rng(
                        stable_seed('e59', seed, mname, q, mi))
                    A_m = perturb_graph(A, ['del', 'add', 'rew'][mi % 3],
                                        q, rng)
                    sm = topk_sets(fn(w['X_anom'], ctx0.with_graph(A_m)))
                    ovs.append(np.mean([len(x & y) / 3
                                        for x, y in zip(s0, sm)]))
                rnd[q] = float(np.mean(ovs))

            # 贪心对抗
            adv = {}
            for B in (2, 4):
                A_cur = A.copy()
                for step in range(B):
                    best, best_d = None, -1
                    for op in candidates(A_cur):
                        A_try = apply_op(A_cur, op)
                        st = topk_sets(fn(w['X_anom'],
                                          ctx0.with_graph(A_try)))
                        dist = mean_jac_dist(s0, st)
                        if dist > best_d:
                            best_d, best = dist, op
                    A_cur = apply_op(A_cur, best)
                s_adv = topk_sets(fn(w['X_anom'], ctx0.with_graph(A_cur)))
                adv[B] = float(np.mean([len(x & y) / 3
                                        for x, y in zip(s0, s_adv)]))

            out.append(dict(seed=seed, method=mname,
                            hit3=float(M.mean_hit(phi0, w['R_anom'], 3)),
                            acr_rand_q01=rnd[0.1], acr_rand_q02=rnd[0.2],
                            acr_adv_B2=adv[2], acr_adv_B4=adv[4]))
            print(out[-1], flush=True)

    json.dump(out, open(os.path.join(CACHE, 'e59_adversarial.json'), 'w'),
              default=float)
    print('\n=== 均值 (5 seeds) ===')
    for m in ('PropRank', 'DPTA-G'):
        sub = [r for r in out if r['method'] == m]
        print(f"{m}: rand(q=.1)={np.mean([r['acr_rand_q01'] for r in sub]):.3f} "
              f"rand(q=.2)={np.mean([r['acr_rand_q02'] for r in sub]):.3f} "
              f"adv(B=2)={np.mean([r['acr_adv_B2'] for r in sub]):.3f} "
              f"adv(B=4)={np.mean([r['acr_adv_B4'] for r in sub]):.3f}")


if __name__ == '__main__':
    run()
