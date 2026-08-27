# -*- coding: utf-8 -*-
"""E60: 创新2 — Prop 2 (必要侧翻转条件) 的经验刻画: 裕度-风险曲线.
SCM medium 5 seeds, true 图, PropRank/DPTA-G, M=8 (q=0.1 循环).
对每窗口每实例计算有符号裕度:
  m_inst = min_{v in T_K} (s_v + eps_v)  -  max_{u not in T_K} (s_u + eps_u)
  (实例扰动后的第K名边界裕度; < 0 ⟺ 该实例翻转)
Prop1 充分条件用 |eps|<=eta 的最坏情形; Prop2 观察:
  翻转当且仅当存在实例 m_inst < 0 (恒真式), 其可部署代理为
  基础裕度 gamma 与位移的相对强度 r = 2*eta_med_est / gamma:
  输出 risk(r) 分箱曲线 + AUC(r -> flip) —— 必要侧的经验校准.
输出: _cache/e60_prop2.json
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
Q = 0.1
M_INST = 8
N_EST = 4


def signed_margin(phi0_row, phim_row, K=3):
    order0 = np.argsort(-phi0_row)
    inside = order0[:K]
    outside = order0[K:]
    m_in = np.min(phim_row[inside])
    m_out = np.max(phim_row[outside])
    return float(m_in - m_out)


def run():
    rows = []
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
            phis = []
            for mi in range(M_INST):
                rng = np.random.default_rng(stable_seed('e60', seed,
                                                        mname, mi))
                A_m = perturb_graph(A, ['del', 'add', 'rew'][mi % 3],
                                    Q, rng)
                phis.append(fn(w['X_anom'], ctx0.with_graph(A_m)))
            phis = np.stack(phis)
            order = np.argsort(-phi0, axis=1)
            gap = phi0[np.arange(len(phi0)), order[:, 2]] - \
                phi0[np.arange(len(phi0)), order[:, 3]]
            eta = np.abs(phis - phi0[None]).max(axis=2).T   # (n, M)
            margins_eval = np.array([[signed_margin(phi0[i], phis[m, i])
                                      for m in range(M_INST)]
                                     for i in range(len(phi0))])
            flip_eval = (margins_eval[:, N_EST:] < 0).mean(axis=1)
            r_ratio = 2 * np.median(eta[:, :N_EST], axis=1) / \
                np.maximum(gap, 1e-12)
            # 评估半最小裕度 (负=翻转)
            min_margin_eval = margins_eval[:, N_EST:].min(axis=1)
            for i in range(len(phi0)):
                rows.append(dict(
                    seed=seed, method=mname, window=i,
                    gap=float(gap[i]), r=float(r_ratio[i]),
                    flip_eval=float(flip_eval[i]),
                    min_margin=float(min_margin_eval[i])))
        print(f'seed {seed} done', flush=True)

    json.dump(rows, open(os.path.join(CACHE, 'e60_prop2.json'), 'w'),
              default=float)
    print('\n=== 裕度-风险曲线 (比值 r = 2*eta_med/gamma) ===')
    from sklearn.metrics import roc_auc_score
    for m in ('PropRank', 'DPTA-G'):
        sub = [r for r in rows if r['method'] == m]
        r = np.array([x['r'] for x in sub])
        f = np.array([x['flip_eval'] for x in sub])
        auc = roc_auc_score(f > 0, r)
        print(f'{m}: n={len(sub)} AUC(r→flip)={auc:.3f} '
              f'翻转率={f.mean():.3f}')
        edges = [0, 0.5, 1.0, 2.0, np.inf]
        for a, b in zip(edges[:-1], edges[1:]):
            msk = (r >= a) & (r < b)
            if msk.sum() > 3:
                print(f'  r∈[{a},{b}): n={msk.sum()} 翻转率={f[msk].mean():.3f}')


if __name__ == '__main__':
    run()
