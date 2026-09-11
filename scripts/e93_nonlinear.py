# -*- coding: utf-8 -*-
"""E93: 非线性 SCM 扩展 (阶段一实验3).

问题: 主网格 SCM 是线性 VAR, 论文自认"模型匹配=传输幅度上界".
设计: 同 15 节点 DAG 拓扑, 结构方程换 tanh 非线性耦合
  x(t+1) = tanh(W x(t)) + eps,  deviational 注入 (medium).
图源: 真支撑图 + 线性 PCMCI 在非线性数据上的学习图(天然有偏).
被试: DPTA-G / PropRank / GraphGranger / zDev / AERec / Random.
输出: hit@3(真图/学习图), ACR@3(del/add/rew 0.2, M=8), 窗口级SW
-> _cache/e93_nonlinear.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graph_free import GRAPH_FREE
from evaluation import metrics as M
from graphs.sources import pcmci_graph
from scorers import make_scorer
from scripts.e1_anchor import perturb_graph, stable_seed

CACHE = os.path.join(SRC, '_cache')
D, T, WIN, STRIDE, MAG, NOISE = 15, 6000, 16, 20, 1.2, 0.5
N_EPISODES = 5


def make_dag(seed):
    rng = np.random.default_rng(seed)
    A_bin = np.zeros((D, D), dtype=float)
    order = rng.permutation(D)
    for i in range(1, D):
        k = min(int(rng.integers(1, 3)), i)
        parents = rng.choice(order[:i], size=k, replace=False)
        A_bin[parents, order[i]] = 1.0
    return A_bin, order


def simulate(A_bin, seed, inject=None):
    """x(t+1) = tanh(W x(t)) + eps; W = 0.6 * A_bin (行归一)."""
    rng = np.random.default_rng(seed + 77)
    W = 0.6 * A_bin / np.maximum(A_bin.sum(0, keepdims=True), 1)
    X = np.zeros((T, D), dtype=np.float32)
    for t in range(T - 1):
        X[t + 1] = np.tanh(W @ X[t]) + rng.normal(0, NOISE, D)
    if inject is not None:
        node, t0, dur = inject
        scale = np.std(X[:t0 - 10, node]) + 1e-6
        for t in range(t0, min(t0 + dur, T)):
            X[t, node] += MAG * scale
    return X


def build(seed):
    A_bin, order = make_dag(seed)
    t0 = T // 2
    ep_len = N_EPISODES * WIN * STRIDE // N_EPISODES  # 每集 16*20=320
    roots, Xa = [], []
    pool_series = simulate(A_bin, seed)
    pool_std = pool_series[:t0 - 10].std(0) + 1e-6
    for e in range(N_EPISODES):
        node = int(order[e])
        s = t0 + e * (WIN * STRIDE)
        X = simulate(A_bin, seed, inject=(node, s, WIN * STRIDE))
        for k in range(WIN):
            w0 = s + k * STRIDE
            Xa.append(X[w0:w0 + WIN])
            roots.append({node})
    X_anom = np.stack(Xa).astype(np.float32)
    mu = pool_series[:t0 - 10].mean(0)
    sd = pool_std
    X_anom = (X_anom - mu) / sd
    rng = np.random.default_rng(seed + 5)
    pool_w = []
    starts = rng.integers(0, t0 - WIN - 1, 400)
    for s0 in starts:
        pool_w.append((pool_series[s0:s0 + WIN] - mu) / sd)
    X_pool = np.stack(pool_w).astype(np.float32)
    return X_anom, roots, X_pool, pool_series[:t0], A_bin


def main():
    out = {'per_seed': [], 'methods': {}}
    acc = {}
    for seed in range(5):
        X_anom, roots, X_pool, pool_series, A_true = build(seed)
        scorer = make_scorer('iforest').fit(X_pool)
        ctx0 = Context(X_pool, scorer=scorer)
        A_pcmci = np.asarray(pcmci_graph(pool_series[:3000], tau_max=1)
                             ['adj'], dtype=float)
        for mname, fn in list(GRAPH_DEPENDENT.items()) + \
                [('zDev', GRAPH_FREE['zDev']),
                 ('AERec', GRAPH_FREE['AERec']),
                 ('Random', GRAPH_FREE['Random'])]:
            for gname, A in (('true', A_true), ('pcmci', A_pcmci)):
                if mname in ('zDev', 'AERec', 'Random') and gname != 'true':
                    continue
                if mname in GRAPH_DEPENDENT:
                    phi0 = fn(X_anom, ctx0.with_graph(A))
                else:
                    phi0 = fn(X_anom, ctx0)
                h = M.hit_at_k(phi0, roots, 3)
                cons = []
                for mi in range(8):
                    rng = np.random.default_rng(
                        stable_seed('e93', seed, mname, gname, mi))
                    if mname in GRAPH_DEPENDENT:
                        A_m = perturb_graph(A, ['del', 'add', 'rew'][mi % 3],
                                            0.2, rng)
                        phi_m = fn(X_anom, ctx0.with_graph(A_m))
                    else:
                        phi_m = fn(X_anom, ctx0)
                    cons.append(M.acr_at_k(phi_m, phi0, 3))
                c = np.stack(cons).mean(0)
                acc.setdefault((mname, gname), []).append(
                    (float(np.mean(h)), float(np.mean(c)),
                     float(((c > 0.8) & (h == 0)).mean())))
        print(f'seed {seed} done', flush=True)
    for (m, g), vals in acc.items():
        out['methods'][f'{m}|{g}'] = dict(
            hit3=float(np.mean([v[0] for v in vals])),
            acr3=float(np.mean([v[1] for v in vals])),
            sw=float(np.mean([v[2] for v in vals])))
        print(f'{m:12s}|{g:6s}', out['methods'][f'{m}|{g}'], flush=True)
    json.dump(out, open(os.path.join(CACHE, 'e93_nonlinear.json'), 'w'),
              indent=1)
    print('-> e93_nonlinear.json')


if __name__ == '__main__':
    main()
