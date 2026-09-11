# -*- coding: utf-8 -*-
"""E5: Round-1 审稿补强实验 (M1/M2/M5/M6)。

M1  关系型异常注入: 对指定边 (i->j) 的 VAR 系数在异常期内漂移, 根因 = j
    (生成机制被改变的节点, 非加性偏差)。检验象限结论是否只在偏差型注入下成立。
M2  TE 根因映射敏感性: 窄映射 (当前) vs 宽映射 (根因 + PCMCI 图一步下游中
    注入后 z>3 的变量)。
M5  DPTA-G 融合权重 α 扫描 (0/.3/.5/.7/1): 伪稳定性是否参数化的假象。
M6  跨数据集图迁移: scm_i 的 PCMCI 学图用于 scm_j (i!=j), 第二系统偏差源。
输出: _cache/e5_relation.json / e5_te_mapping.json / e5_alpha.json / e5_transfer.json
"""
import os
import sys
import json
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.scm import (sample_random_dag, dag_to_lag_coefs, gen_var_series,
                      gold_edges_from_coefs, build_windows)
from data.tep import load_tep_units
from graphs.sources import pcmci_graph
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT, _robust_z, GraphRegression
from attribution.graph_free import GRAPH_FREE
from evaluation import metrics as M
from scripts.e1_anchor import perturb_graph, stable_seed, M_PERTURB

CACHE = os.path.join(SRC, '_cache')


# ---------------------------------------------------------------- M1
def make_relation_sample(n_nodes=15, edge_prob=0.2, tau_max=3, T=5000,
                         n_anomalies=7, seed=0, noise_std=0.5,
                         coef_drift=2.5):
    """关系型异常: 异常期内指定边 (src->tgt) 的全部 lag 系数乘 coef_drift,
    根因 = tgt (其生成关系被改变; 边际分布变化为二阶效应)。"""
    A, G = sample_random_dag(n_nodes, edge_prob, seed=seed)
    coefs = dag_to_lag_coefs(A, tau_max=tau_max)
    gold = gold_edges_from_coefs(coefs)
    rng = np.random.default_rng(seed + 1000)
    edges = sorted(gold)
    anomalies = []
    series = gen_var_series(coefs, T=T, seed=seed, noise_std=noise_std)
    anom_mask = np.zeros(T, dtype=bool)
    placed = []
    for k in range(n_anomalies):
        # 选一条边, 目标节点尽量有下游
        for _ in range(50):
            src, tgt = edges[rng.integers(len(edges))]
            if all(abs(tgt - p) > 300 for p in placed):
                break
        t0 = 600 + k * 550
        placed.append(t0)
        # 重生成该段: 异常期用漂移系数
        drift = {key: c * coef_drift for key, c in coefs.items()
                 if key[1] == src and key[0] == tgt}
        seg_coefs = dict(coefs)
        seg_coefs.update(drift)
        # 手工前向模拟异常段 (覆盖 series)
        tau_max_ = tau_max
        data = series
        by_target = collections.defaultdict(list)
        for (t_, s_, l_), c in seg_coefs.items():
            by_target[t_].append((s_, l_, c))
        for t in range(t0, min(t0 + 200, T)):
            x = data[t].copy()
            for t_, lst in by_target.items():
                acc = 0.0
                for s_, l_, c in lst:
                    acc += c * data[t - l_, s_]
                x[t_] = acc
            # 全部节点保留每步噪声项 (与正常段同构); 否则异常段出现"方差收缩"
            # 伪影 —— 非根因节点方差骤降会被重构类方法当作信号
            x += rng.normal(0, noise_std, n_nodes)
            data[t] = x
        anom_mask[t0:t0 + 200] = True
        anomalies.append(dict(t_start=t0, length=200, kind='relation',
                              roots=[tgt], propagation=sorted(
                                  __import__('networkx').descendants(G, tgt))))
    sample = dict(normal=gen_var_series(coefs, T=T, seed=seed,
                                        noise_std=noise_std),
                  series=series, adjacency=A, graph=G,
                  lag_coefs=coefs, gold_edges=gold, anomalies=anomalies,
                  n_nodes=n_nodes, T=T, seed=seed)
    return sample


def run_m1():
    records = []
    for seed in range(5):
        s = make_relation_sample(seed=seed)
        w = build_windows(s)
        A_true = s['adjacency'].astype(float)
        A_clean = pcmci_graph(s['normal'])['adj']
        A_anom = pcmci_graph(s['series'])['adj']
        for sk in ('iforest', 'pca'):
            scorer = make_scorer(sk).fit(w['X_pool'])
            ctx0 = Context(w['X_pool'], scorer=scorer)
            if sk == 'pca':
                from scorers import TorchPCAScorer
                ctx0.torch_scorer = TorchPCAScorer(scorer)
            for mname, fn in list(GRAPH_DEPENDENT.items()):
                for sname, A in (('true', A_true), ('clean', A_clean),
                                 ('anom', A_anom)):
                    phi = fn(w['X_anom'], ctx0.with_graph(A))
                    r = dict(exp='m1_relation', dataset=f'scm{seed}',
                             scorer=sk, method=mname, graph_source=sname,
                             hit3=float(M.mean_hit(phi, w['R_anom'], 3)),
                             hit1=float(M.mean_hit(phi, w['R_anom'], 1)))
                    # ACR(部署)
                    acrs = []
                    for mi in range(M_PERTURB):
                        rng = np.random.default_rng(stable_seed(
                            'm1', seed, sk, mname, sname, mi))
                        A_m = perturb_graph(A, ['del', 'add', 'rew'][mi % 3],
                                            0.2, rng)
                        phi_m = fn(w['X_anom'], ctx0.with_graph(A_m))
                        acrs.append(M.acr_at_k(phi_m, phi, 3).mean())
                    r['acr3'] = float(np.mean(acrs))
                    records.append(r)
            for mname, fn in GRAPH_FREE.items():
                if mname == 'Grad' and sk != 'pca':
                    continue
                phi = fn(w['X_anom'], ctx0)
                records.append(dict(exp='m1_relation', dataset=f'scm{seed}',
                                    scorer=sk, method=mname, graph_source='none',
                                    hit3=float(M.mean_hit(phi, w['R_anom'], 3)),
                                    hit1=float(M.mean_hit(phi, w['R_anom'], 1)),
                                    acr3=None))
        print(f"[M1] seed{seed} done", flush=True)
    json.dump(records, open(os.path.join(CACHE, 'e5_relation.json'), 'w'),
              ensure_ascii=False, default=float)
    # 摘要
    agg = collections.defaultdict(list)
    for r in records:
        agg[(r['method'], r['graph_source'])].append(r['hit3'])
    print('=== M1 relation-anomaly hit@3 (method, source) ===')
    for k in sorted(agg):
        print(f'  {k}: {np.mean(agg[k]):.3f}')
    return records


# ---------------------------------------------------------------- M2
def run_m2():
    units = load_tep_units()
    records = []
    for u in units:
        cache = os.path.join(CACHE, f"te_graph_{u['name']}.npz")
        A = np.load(cache, allow_pickle=True)['A']
        for sk in ('iforest', 'pca', 'ocsvm'):
            scorer = make_scorer(sk).fit(u['X_pool'])
            ctx0 = Context(u['X_pool'], scorer=scorer)
            st = ctx0.pool_stats
            roots_narrow = next(iter(u['R_anom']))
            # 宽映射: 根因 + 图一步下游中注入后 z>3 的变量
            children = set(np.where(A[list(roots_narrow)[0]] > 0)[0])
            z_anom = np.abs(u['X_anom'].mean(1) - st['med']) / st['mad']
            resp = set(np.where(z_anom.mean(0) > 3.0)[0])
            roots_wide = roots_narrow | (children & resp)
            for mname, fn in list(GRAPH_DEPENDENT.items()) + \
                    [(k, v) for k, v in GRAPH_FREE.items()
                     if k not in ('Grad',)]:
                ctx = ctx0.with_graph(A) if mname in GRAPH_DEPENDENT else ctx0
                X = u['X_anom']
                # phi 一次, 两种金标准各算 hit
                phi = fn(X, ctx)
                if mname == 'Grad' and sk != 'pca':
                    continue
                records.append(dict(
                    exp='m2_te_mapping', dataset=u['name'], scorer=sk,
                    method=mname,
                    hit3_narrow=float(M.mean_hit(phi, u['R_anom'], 3)),
                    hit3_wide=float(M.mean_hit(
                        phi, [roots_wide] * len(X), 3)),
                    n_wide=len(roots_wide)))
        print(f"[M2] {u['name']} wide-roots={sorted(roots_wide)}", flush=True)
    json.dump(records, open(os.path.join(CACHE, 'e5_te_mapping.json'), 'w'),
              ensure_ascii=False, default=float)
    agg = collections.defaultdict(lambda: dict(n=[], w=[]))
    for r in records:
        agg[r['method']]['n'].append(r['hit3_narrow'])
        agg[r['method']]['w'].append(r['hit3_wide'])
    print('=== M2 TE mapping sensitivity (narrow -> wide) ===')
    for m in sorted(agg):
        print(f"  {m:14s} {np.mean(agg[m]['n']):.3f} -> {np.mean(agg[m]['w']):.3f}")
    return records


# ---------------------------------------------------------------- M5
def run_m5():
    from attribution import normalize_rows

    def dpta_alpha(X, ctx, alpha):
        z = _robust_z(X, ctx)
        rel = GraphRegression(ctx).residual_z(X, ctx.graph)
        nz, nr = normalize_rows(z), normalize_rows(rel)
        return alpha * nz + (1 - alpha) * nr

    records = []
    for seed in range(5):
        s = __import__('data.scm', fromlist=['make_sample']).make_sample(
            n_nodes=15, T=5000, n_single=5, n_joint=2, seed=seed,
            noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        A_true = s['adjacency'].astype(float)
        A_anom = pcmci_graph(s['series'])['adj']
        for sk in ('iforest',):
            scorer = make_scorer(sk).fit(w['X_pool'])
            ctx0 = Context(w['X_pool'], scorer=scorer)
            for alpha in (0.0, 0.3, 0.5, 0.7, 1.0):
                for sname, A in (('true', A_true), ('anom', A_anom)):
                    phi = dpta_alpha(w['X_anom'], ctx0.with_graph(A), alpha)
                    r = dict(exp='m5_alpha', dataset=f'scm{seed}', scorer=sk,
                             alpha=alpha, graph_source=sname,
                             hit3=float(M.mean_hit(phi, w['R_anom'], 3)))
                    acrs = []
                    phi_t = dpta_alpha(w['X_anom'],
                                       ctx0.with_graph(A_true), alpha)
                    for mi in range(M_PERTURB):
                        rng = np.random.default_rng(stable_seed(
                            'm5', seed, alpha, sname, mi))
                        A_m = perturb_graph(A, ['del', 'add', 'rew'][mi % 3],
                                            0.2, rng)
                        phi_m = dpta_alpha(w['X_anom'],
                                           ctx0.with_graph(A_m), alpha)
                        acrs.append(M.acr_at_k(phi_m, phi, 3).mean())
                    r['acr3'] = float(np.mean(acrs))
                    records.append(r)
        print(f"[M5] seed{seed} done", flush=True)
    json.dump(records, open(os.path.join(CACHE, 'e5_alpha.json'), 'w'),
              ensure_ascii=False, default=float)
    agg = collections.defaultdict(list)
    for r in records:
        agg[(r['alpha'], r['graph_source'])].append((r['hit3'], r['acr3']))
    print('=== M5 alpha sweep (alpha, source): (hit3, acr3) ===')
    for k in sorted(agg):
        h = np.mean([a for a, _ in agg[k]])
        a = np.mean([b for _, b in agg[k]])
        print(f'  alpha={k[0]} src={k[1]:5s} hit={h:.3f} acr={a:.3f}')
    return records


# ---------------------------------------------------------------- M6
def run_m6():
    samples = []
    for seed in range(5):
        s = __import__('data.scm', fromlist=['make_sample']).make_sample(
            n_nodes=15, T=5000, n_single=5, n_joint=2, seed=seed,
            noise_std=0.5, magnitude=0.5)
        samples.append((seed, s, pcmci_graph(s['normal'])['adj']))
    records = []
    for tgt_seed, s, _ in samples:
        w = build_windows(s)
        scorer = make_scorer('iforest').fit(w['X_pool'])
        ctx0 = Context(w['X_pool'], scorer=scorer)
        for mname in ('PropRank', 'DPTA-G'):
            fn = GRAPH_DEPENDENT[mname]
            for src_seed, _, A_src in samples:
                if src_seed == tgt_seed:
                    continue
                phi = fn(w['X_anom'], ctx0.with_graph(A_src))
                records.append(dict(
                    exp='m6_transfer', dataset=f'scm{tgt_seed}',
                    graph_from=f'scm{src_seed}', method=mname,
                    hit3=float(M.mean_hit(phi, w['R_anom'], 3))))
        # 自图基线
        _, _, A_self = samples[tgt_seed]
        for mname in ('PropRank', 'DPTA-G'):
            fn = GRAPH_DEPENDENT[mname]
            phi = fn(w['X_anom'], ctx0.with_graph(A_self))
            records.append(dict(
                exp='m6_transfer', dataset=f'scm{tgt_seed}',
                graph_from='self', method=mname,
                hit3=float(M.mean_hit(phi, w['R_anom'], 3))))
        print(f"[M6] scm{tgt_seed} done", flush=True)
    json.dump(records, open(os.path.join(CACHE, 'e5_transfer.json'), 'w'),
              ensure_ascii=False, default=float)
    agg = collections.defaultdict(list)
    for r in records:
        kind = 'self' if r['graph_from'] == 'self' else 'transfer'
        agg[(r['method'], kind)].append(r['hit3'])
    print('=== M6 graph transfer (method, kind): hit3 ===')
    for k in sorted(agg):
        print(f'  {k}: {np.mean(agg[k]):.3f}')
    return records


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', 'm5'):
        run_m5()
    if which in ('all', 'm1'):
        run_m1()
    if which in ('all', 'm6'):
        run_m6()
    if which in ('all', 'm2'):
        run_m2()
