# -*- coding: utf-8 -*-
"""E11 (方案2/3/4/6): 主动边验证 + 稳定性证书 + 共识集 + α诱导图源."""
import os
import sys
import json
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.scm import make_sample, build_windows
from graphs.sources import pcmci_graph
from graphs import graph_ops as go
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT, _robust_z, GraphRegression
from evaluation import metrics as M
from scripts.e1_anchor import perturb_graph, stable_seed, M_PERTURB

CACHE = os.path.join(SRC, '_cache')


# ================= 方案2: 主动边验证 =================
def p2_active_verification():
    """系统偏差(污染图)下: 贪心(按边对phi的影响度) vs 随机 验证 k 条边.
    验证动作 = 从图中删除被确认为假的边; 影响度 = 删除该边后 phi 排序变化量
    在窗口集上的平均."""
    records = []
    for seed in range(5):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2, seed=seed,
                        noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        A_true = s['adjacency'].astype(float)
        A_bad = pcmci_graph(s['series'])['adj']     # 污染图 (系统偏差源)
        scorer = make_scorer('iforest').fit(w['X_pool'])
        ctx0 = Context(w['X_pool'], scorer=scorer)
        for mname in ('PropRank', 'DPTA-G'):
            fn = GRAPH_DEPENDENT[mname]
            phi_target = fn(w['X_anom'], ctx0.with_graph(A_true))
            oracle = float(M.mean_hit(phi_target, w['R_anom'], 3))
            phi_bad = fn(w['X_anom'], ctx0.with_graph(A_bad))
            base = float(M.mean_hit(phi_bad, w['R_anom'], 3))
            # 边影响度: 逐边删除 -> phi 排序变化 (ACR 意义)
            edges = go.edges_from_adj(A_bad)
            infl = {}
            for (i, j) in edges:
                A_e = A_bad.copy()
                A_e[i, j] = 0.0
                phi_e = fn(w['X_anom'], ctx0.with_graph(A_e))
                infl[(i, j)] = 1.0 - float(M.acr_at_k(phi_e, phi_bad, 3).mean())
            ranked = [e for e, _ in sorted(infl.items(), key=lambda kv: -kv[1])]
            rng = np.random.default_rng(seed)
            for k in (1, 2, 3, 5, 8, 12):
                # 贪心: 删除影响度最高的 k 条 (被验证为假)
                A_g = A_bad.copy()
                for e in ranked[:k]:
                    A_g[e[0], e[1]] = 0.0
                hit_g = float(M.mean_hit(
                    fn(w['X_anom'], ctx0.with_graph(A_g)), w['R_anom'], 3))
                # 随机对照 (5 次平均)
                hits_r = []
                for t in range(5):
                    A_r = A_bad.copy()
                    for e in rng.choice(len(edges), size=min(k, len(edges)),
                                        replace=False):
                        A_r[edges[e][0], edges[e][1]] = 0.0
                    hits_r.append(float(M.mean_hit(
                        fn(w['X_anom'], ctx0.with_graph(A_r)), w['R_anom'], 3)))
                records.append(dict(exp='p2', dataset=f'scm{seed}', method=mname,
                                    k=k, hit_base=base, hit_greedy=hit_g,
                                    hit_random=float(np.mean(hits_r)),
                                    hit_oracle=oracle))
        print(f'[P2] scm{seed} done', flush=True)
    json.dump(records, open(os.path.join(CACHE, 'e11_p2.json'), 'w'),
              default=float)
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in records:
        agg[(r['method'], r['k'])]['g'].append(r['hit_greedy'])
        agg[(r['method'], r['k'])]['r'].append(r['hit_random'])
        agg[(r['method'], r['k'])]['b'].append(r['hit_base'])
        agg[(r['method'], r['k'])]['o'].append(r['hit_oracle'])
    print('=== P2: active edge verification (method, k): base->greedy/random (oracle) ===')
    for k_ in sorted(agg, key=str):
        v = agg[k_]
        print(f'  {str(k_):24s} {np.mean(v["b"]):.3f} -> {np.mean(v["g"]):.3f} /'
              f' {np.mean(v["r"]):.3f}  (oracle {np.mean(v["o"]):.3f})')


# ================= 方案3: 稳定性证书 =================
def p3_certificate():
    """证书: 若 phi 的 top3/第4名间隔 > 扰动引起的 phi 变化上界(经验), 则
    扰动下 top3 必不变. 统计持证窗口中 stable-wrong 的占比."""
    records = []
    for seed in range(5):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2, seed=seed,
                        noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        A = s['adjacency'].astype(float)
        scorer = make_scorer('iforest').fit(w['X_pool'])
        ctx0 = Context(w['X_pool'], scorer=scorer)
        for mname in ('PropRank', 'DPTA-G'):
            fn = GRAPH_DEPENDENT[mname]
            phi0 = fn(w['X_anom'], ctx0.with_graph(A))
            hits = M.hit_at_k(phi0, w['R_anom'], 3)
            # 扰动 phi (8 个) 与每窗口经验扰动幅度 delta_i = max_j |phi_m - phi_0|
            deltas = []
            phis = []
            for mi in range(8):
                rng = np.random.default_rng(stable_seed('p3', seed, mname, mi))
                A_m = perturb_graph(A, ['del', 'add', 'rew'][mi % 3], 0.2, rng)
                phis.append(fn(w['X_anom'], ctx0.with_graph(A_m)))
            D = np.stack(phis) - phi0[None]          # (8, n, d)
            delta = np.abs(D).max(0)                  # (n, d) 每变量的最大偏移
            order = np.argsort(-phi0, axis=1)
            for i in range(len(phi0)):
                gap = phi0[i, order[i, 2]] - phi0[i, order[i, 3]]   # top3 vs 第4
                # 充分条件: 全变量最大位移 delta_all; gap > 2*delta_all 时
                # 任何非成员不可能超过任何成员 -> top-3 集合不变
                delta_all = delta[i].max()
                certified = gap > 2 * delta_all
                records.append(dict(exp='p3', dataset=f'scm{seed}', method=mname,
                                    window=i, certified=bool(certified),
                                    hit=int(hits[i])))
    json.dump(records, open(os.path.join(CACHE, 'e11_p3.json'), 'w'))
    agg = collections.defaultdict(list)
    for r in records:
        agg[(r['method'], r['certified'])].append(r['hit'])
    print('=== P3: certificate coverage & validity ===')
    for k_ in sorted(agg, key=str):
        v = agg[k_]
        print(f'  {str(k_):28s} n={len(v)} hit@3={np.mean(v):.3f}')


# ================= 方案4: 共识归因集 =================
def p4_consensus():
    """共识集 = 扰动下以频率>=q 出现在 top3 的变量集; 集大小与命中关系."""
    records = []
    for seed in range(5):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2, seed=seed,
                        noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        A = s['adjacency'].astype(float)
        scorer = make_scorer('iforest').fit(w['X_pool'])
        ctx0 = Context(w['X_pool'], scorer=scorer)
        for mname in ('PropRank', 'DPTA-G', 'GraphGranger'):
            fn = GRAPH_DEPENDENT[mname]
            phi0 = fn(w['X_anom'], ctx0.with_graph(A))
            phis = [phi0]
            for mi in range(8):
                rng = np.random.default_rng(stable_seed('p4', seed, mname, mi))
                A_m = perturb_graph(A, ['del', 'add', 'rew'][mi % 3], 0.2, rng)
                phis.append(fn(w['X_anom'], ctx0.with_graph(A_m)))
            for i in range(len(phi0)):
                from evaluation.metrics import topk
                cnt = collections.Counter()
                for p in phis:
                    for v in topk(p[i], 3):
                        cnt[v] += 1
                for q in (0.5, 0.8):
                    cset = {v for v, c in cnt.items() if c / len(phis) >= q}
                    records.append(dict(exp='p4', dataset=f'scm{seed}',
                                        method=mname, window=i, q=q,
                                        size=len(cset),
                                        hit=int(bool(cset & set(w['R_anom'][i])))))
    json.dump(records, open(os.path.join(CACHE, 'e11_p4.json'), 'w'))
    agg = collections.defaultdict(list)
    for r in records:
        agg[(r['method'], r['q'], r['size'])].append(r['hit'])
    print('=== P4: consensus set size -> hit (method, q, |set|) ===')
    for k_ in sorted(agg, key=str):
        v = agg[k_]
        print(f'  {str(k_):30s} n={len(v)} hit={np.mean(v):.2f}')


# ================= 方案6: α 诱导图源 =================
def p6_alpha_sources():
    """pc_alpha ∈ {0.01, 0.05, 0.2} -> 图质量/密度 -> PropRank/DPTA-G hit."""
    records = []
    for seed in range(5):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2, seed=seed,
                        noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        A_true = s['adjacency'].astype(float)
        scorer = make_scorer('iforest').fit(w['X_pool'])
        ctx0 = Context(w['X_pool'], scorer=scorer)
        for alpha in (0.01, 0.05, 0.2):
            A_a = pcmci_graph(s['normal'], pc_alpha=alpha)['adj']
            rec, prec = go.edge_recall_precision(A_a, A_true)
            for mname in ('PropRank', 'DPTA-G', 'GCN-Rank'):
                from scripts.e8_round6 import gcn_rank_attribute
                fn = (GRAPH_DEPENDENT.get(mname) or gcn_rank_attribute)
                phi = fn(w['X_anom'], ctx0.with_graph(A_a))
                records.append(dict(exp='p6', dataset=f'scm{seed}', alpha=alpha,
                                    method=mname, n_edges=int(A_a.sum()),
                                    recall=rec, precision=prec,
                                    hit3=float(M.mean_hit(phi, w['R_anom'], 3))))
        print(f'[P6] scm{seed} done', flush=True)
    json.dump(records, open(os.path.join(CACHE, 'e11_p6.json'), 'w'),
              default=float)
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in records:
        agg[(r['method'], r['alpha'])]['h'].append(r['hit3'])
        agg[(r['method'], r['alpha'])]['e'].append(r['n_edges'])
        agg[(r['method'], r['alpha'])]['p'].append(r['precision'])
    print('=== P6: alpha-induced graph bias (method, alpha): edges/prec -> hit3 ===')
    for k_ in sorted(agg, key=str):
        v = agg[k_]
        print(f'  {str(k_):26s} E={np.mean(v["e"]):5.1f} prec={np.mean(v["p"]):.2f}'
              f' hit={np.mean(v["h"]):.3f}')


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', 'p2'):
        p2_active_verification()
    if which in ('all', 'p3'):
        p3_certificate()
    if which in ('all', 'p4'):
        p4_consensus()
    if which in ('all', 'p6'):
        p6_alpha_sources()
