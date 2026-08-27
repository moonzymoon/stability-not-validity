# -*- coding: utf-8 -*-
"""E4 缓解实验: 扰动集成 / 多图投票 / 置信度加权 (v2: 分随机 vs 系统性偏差两场景)。

预期结论 (待实测, 不预设): 集成挽随机误差, 不修系统偏差。
场景A (随机):   base=G_true,  集成 = {扰动图 phi} 平均        vs 单图 phi(G_true)
场景B (系统性): base=G_anom,  集成 = {G_anom 的扰动 phi} 平均  vs 单图 phi(G_anom)
              + 上限对照 phi(G_true) (oracle)
加边置信度加权: 集成权重 ∝ 图的平均 PCMCI |val| (污染图低置信边多 → 权重低)。
输出: _cache/e4_mitigation.json
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.scm import make_sample, build_windows
from graphs.sources import pcmci_graph
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from evaluation import metrics as M
from scripts.e1_anchor import perturb_graph, stable_seed

CACHE = os.path.join(SRC, '_cache')
M_ENS = 8


def rank_avg(phis):
    """rank-based ensemble: midrank 处理并列 (scipy rankdata)。"""
    from scipy.stats import rankdata
    out = np.zeros_like(phis[0])
    for p in phis:
        for i in range(len(p)):
            r = rankdata(p[i], method='average')
            out[i] += (r - 1) / (len(p[i]) - 1)
    return out / len(phis)


def run():
    records = []
    for seed in range(5):
        for mag in (0.25, 0.5):
            s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2, seed=seed,
                            noise_std=0.5, magnitude=mag)
            w = build_windows(s)
            A_true = s['adjacency'].astype(float)
            A_anom = pcmci_graph(s['series'])['adj']
            A_clean = pcmci_graph(s['normal'])['adj']
            # 边置信度 (PCMCI |val| 最大 lag, 仅用观测值)
            val_c = np.abs(pcmci_graph(s['normal'])['val_matrix']).max(2)

            for sk in ('iforest', 'pca'):
                scorer = make_scorer(sk).fit(w['X_pool'])
                ctx0 = Context(w['X_pool'], scorer=scorer)
                for mname, fn in GRAPH_DEPENDENT.items():
                    for scen, A_base in (('random', A_true), ('systematic', A_anom)):
                        phi_base = fn(w['X_anom'], ctx0.with_graph(A_base))
                        # 扰动图集合 (del/add/rew 混合采样)
                        phis, weights = [], []
                        for mi in range(M_ENS):
                            rng = np.random.default_rng(stable_seed(
                                'ens', seed, mag, sk, mname, scen, mi))
                            fam = ['del', 'del', 'add', 'rew'][mi % 4]
                            st = [0.2, 0.3, 0.2, 0.3][mi % 4]
                            A_m = perturb_graph(A_base, fam, st, rng)
                            phis.append(fn(w['X_anom'], ctx0.with_graph(A_m)))
                            # 图权重: 图内边平均置信度 (只保留图内边)
                            esel = A_m > 0
                            weights.append(float(val_c[esel].mean()) if esel.any() else 1.0)
                        weights = np.array(weights) + 1e-9
                        weights = weights / weights.sum()
                        ens_mean = np.mean(phis, 0)
                        ens_rank = rank_avg(phis)
                        ens_w = sum(w0 * p for w0, p in zip(weights, phis))
                        phi_true = fn(w['X_anom'], ctx0.with_graph(A_true))
                        # random 场景: single = 随机单个损坏图 (实践者图已随机损坏);
                        # systematic 场景: single = 污染学图 (基准 G_base)
                        if scen == 'random':
                            # 同分布对照: single = 从集成相同的 8 个扰动图中随机取 1
                            j = int(np.random.default_rng(stable_seed(
                                'pick', seed, mag, sk, mname, scen)).integers(M_ENS))
                            phi_single = phis[j]
                        else:
                            phi_single = phi_base
                        rec = dict(
                            dataset=f'scm{seed}_m{mag}', scorer=sk, method=mname,
                            scenario=scen,
                            hit_single=float(M.mean_hit(phi_single, w['R_anom'], 3)),
                            hit_ens_mean=float(M.mean_hit(ens_mean, w['R_anom'], 3)),
                            hit_ens_rank=float(M.mean_hit(ens_rank, w['R_anom'], 3)),
                            hit_ens_conf=float(M.mean_hit(ens_w, w['R_anom'], 3)),
                            hit_oracle_true=float(M.mean_hit(phi_true, w['R_anom'], 3)),
                        )
                        records.append(rec)
                        print(f"{rec['dataset']} {sk} {mname} {scen}: "
                              f"single={rec['hit_single']:.2f} "
                              f"ens={rec['hit_ens_mean']:.2f} "
                              f"conf={rec['hit_ens_conf']:.2f} "
                              f"oracle={rec['hit_oracle_true']:.2f}", flush=True)
    out = os.path.join(CACHE, 'e4_mitigation.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=1, default=float)
    print(f"-> {out}")


if __name__ == '__main__':
    run()
