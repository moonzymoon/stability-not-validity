# -*- coding: utf-8 -*-
"""E43: 失败案例走查 — 选一个"稳定但错误"的真实窗口做解剖.

在 SWaT 注入测试床上: PropRank 在相关性图 (corr, 人工/部署可得图源) 上运行,
取 8 个种子扰动下的 ACR 高 (稳定) 且 hit@3=0 的第一个窗口,
输出: 真根因变量、top-5 归因排行与得分、扰动实例的 top-3 一致性、
平稳性证据 (扰动间 top-3 Jaccard), 以及机理判别数据
(扰动不变得分占比 / 真根因在图中的可达性).
输出: _cache/e43_case.json
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.swat import build_swat_units, silver_adj
from scripts.e8_round6 import corr_graph
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from evaluation import metrics as M
from scripts.e1_anchor import perturb_graph, stable_seed

CACHE = os.path.join(SRC, '_cache')


def run():
    u = build_swat_units()
    A_silver = u['silver_adj']
    A_corr = corr_graph(u['series_normal'][:50000], thresh=0.3)
    scorer = make_scorer('iforest').fit(u['X_pool'])
    ctx = Context(u['X_pool'], scorer=scorer).with_graph(A_corr)
    fn = GRAPH_DEPENDENT['PropRank']
    phi0 = fn(u['X_anom'], ctx)
    phis = []
    for mi in range(8):
        rng = np.random.default_rng(stable_seed('e43', mi))
        A_m = perturb_graph(A_corr, ['del', 'add', 'rew'][mi % 3], 0.2, rng)
        phis.append(fn(u['X_anom'], ctx.with_graph(A_m)))
    acr = np.mean([M.acr_at_k(p, phi0, 3) for p in phis], axis=0)
    hits = M.hit_at_k(phi0, u['R_anom'], 3)

    idx = [i for i in range(len(phi0)) if acr[i] == 1.0 and hits[i] == 0]
    i = idx[0]
    root = sorted(u['R_anom'][i])
    order = np.argsort(-phi0[i])
    top5 = [(int(v), round(float(phi0[i, v]), 4)) for v in order[:5]]
    pert_top3 = [sorted(np.argsort(-p[i])[:3].tolist()) for p in phis]
    jac = []
    for p in pert_top3:
        a, b = set(p), set(order[:3].tolist())
        jac.append(len(a & b) / len(a | b))
    # 真根因在 corr 图上从 top1 出发是否可达 (结构错配证据)
    reach_from_top1 = bool(A_corr[top5[0][0], root[0]] > 0 or
                           A_corr[root[0], top5[0][0]] > 0)
    out = dict(
        testbed='SWaT (real process data, injected ground truth)',
        window_index=int(i), true_root=[int(r) for r in root],
        acr_at3=float(acr[i]), hit_at3=int(hits[i]),
        top5_scores=top5,
        perturbation_top3=pert_top3,
        mean_jaccard_vs_reference=float(np.mean(jac)),
        root_linked_to_top1_in_graph=reach_from_top1,
        score_gap_top1_root=float(phi0[i, order[0]] -
                                  phi0[i, root[0]]),
        n_windows_stable_wrong=len(idx),
    )
    json.dump(out, open(os.path.join(CACHE, 'e43_case.json'), 'w'),
              default=float)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    run()
