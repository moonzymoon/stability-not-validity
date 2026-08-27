# -*- coding: utf-8 -*-
"""GRAA v4: Confidence-Pruned PropRank + zDev Ensemble.

核心改进: 用PCMCI边置信度直接剪枝(移除低置信边), 而非逐边一致性评估。
更简单、更激进、计算量小。剪枝后的图更干净 → PropRank更准。
"""
import numpy as np
from . import Context, normalize_rows
from .graph_dep import proprank_attribute
from .graph_free import zdev_attribute


def graa_v4_attribute(X, ctx, edge_conf=None, prune_frac=0.3, gamma=1.0):
    """GRAA v4: Confidence-Pruned PropRank + zDev.

    Parameters:
    -----------
    edge_conf : (D, D) PCMCI |val| 置信度矩阵
    prune_frac : 剪枝比例(移除最低置信的该比例边)
    gamma : zDev混合权重控制
    """
    A = ctx.graph
    n, W, D = X.shape

    # 1. 置信度剪枝
    mean_conf = 1.0
    pruned_conf = 1.0
    if edge_conf is not None:
        edges = [(A[i,j], edge_conf[i,j], i, j)
                 for i in range(D) for j in range(D) if A[i,j] > 0]
        edges.sort(key=lambda x: x[1])  # 按置信度升序 (最低在前)
        n_prune = int(len(edges) * prune_frac)
        A_pruned = A.copy()
        for _, _, i, j in edges[:n_prune]:
            A_pruned[i, j] = 0.0
        mean_conf = np.mean([c for _, c, _, _ in edges])
        retained = [c for _, c, _, _ in edges[n_prune:]]
        # 空保留集 -> 纯 zDev 回退 (pruned_conf=0 => w_prop=0)
        pruned_conf = np.mean(retained) if retained else 0.0
    else:
        A_pruned = A

    # 2. 剪枝图上的PropRank
    ctx_pruned = ctx.with_graph(A_pruned)
    phi_prop = proprank_attribute(X, ctx_pruned)
    nz_prop = normalize_rows(phi_prop)

    # 3. zDev
    phi_zdev = zdev_attribute(X, ctx)
    nz_zdev = normalize_rows(phi_zdev)

    # 4. 按图质量(保留边平均置信度)加权
    # 高置信度 → 信PropRank; 低 → 信zDev
    w_prop = pruned_conf ** gamma  # gamma控制非线性
    phi = w_prop * nz_prop + (1 - w_prop) * nz_zdev

    info = {
        'method': 'GRAA-v4',
        'n_original': int(A.sum()),
        'n_pruned': int(A.sum() - A_pruned.sum()),
        'mean_conf': float(mean_conf),
        'pruned_conf': float(pruned_conf),
        'w_prop': float(w_prop),
    }
    return phi, info
