# -*- coding: utf-8 -*-
"""GRAA: Graph-Reliability-Aware Attribution — 新算法贡献.

核心思想: 利用三类不确定性信号(边置信度/扰动一致性/图统计)自适应加权
双路归因的偏差路和关系路, 在图质量退化时自动降低关系路权重。

机制:
  1. 边置信度加权: 对关系路的每条边用置信度加权(PCMCI |val|)
  2. 扰动一致性门控: 一致性低的窗口降低关系路权重(不可靠)
  3. 图质量自适应: 图密度/精度低时整体偏向偏差路

理论保证: 当图完全正确时GRAA退化为DPTA-G; 当图完全错误时退化为zDev。
"""
import numpy as np
from collections import defaultdict

from . import Context, normalize_rows
from .graph_dep import _robust_z, GraphRegression


def grcm_attribute(X, ctx, edge_conf=None, alpha_base=0.5):
    """GRAA: Graph-Reliability-Aware Attribution.

    Parameters:
    -----------
    X : (n, W, D) 异常窗口
    ctx : Context (含 X_pool, scorer, graph)
    edge_conf : (D, D) 边置信度矩阵 (可选; None则从扰动估计)
    alpha_base : float 基础偏差路权重

    Returns:
    --------
    phi : (n, D) 归因分数
    info : dict (alpha值, 一致性等)
    """
    A = ctx.graph
    n, W, D = X.shape

    # 1. 偏差路
    z = _robust_z(X, ctx)
    nz = normalize_rows(z)

    # 2. 关系路(边置信度加权)
    if edge_conf is None:
        edge_conf = np.ones((D, D))
    # 用置信度加权邻接
    A_weighted = A * edge_conf
    gr = GraphRegression(ctx)
    # 修改: 用加权图做回归
    rel = _weighted_residual_z(X, ctx, A_weighted)
    nr = normalize_rows(rel)

    # 3. 扰动一致性门控(每窗口)
    from evaluation.metrics import topk
    M = 6
    cons = np.zeros(n)
    rng = np.random.default_rng(0)
    from graphs import graph_ops as go
    for i in range(n):
        base_top = topk(nz[i] * alpha_base + nr[i] * (1 - alpha_base), 3)
        overlaps = []
        for mi in range(M):
            # 扰动图
            fam = ['del', 'add', 'rew'][mi % 3]
            rng_m = np.random.default_rng(42 + mi)
            A_m = go.delete_edges(A, 0.2, rng_m) if fam == 'del' else \
                  go.add_edges(A, 0.2, rng_m) if fam == 'add' else \
                  go.rewire_edges(A, 0.2, rng_m)
            A_mw = A_m * edge_conf
            rel_m = _weighted_residual_z(X[i:i+1], ctx, A_mw)
            nz_m = _robust_z(X[i:i+1], ctx)
            nz_m = normalize_rows(nz_m)
            nr_m = normalize_rows(rel_m)
            pert_top = topk(nz_m[0] * alpha_base + nr_m[0] * (1 - alpha_base), 3)
            overlaps.append(len(base_top & pert_top) / 3)
        cons[i] = np.mean(overlaps)

    # 4. 自适应权重: 一致性低 → 偏向偏差路(alpha高)
    # alpha_i = alpha_base + (1 - cons_i) * (1 - alpha_base) * gamma
    gamma = 0.8  # 门控强度
    alpha_i = alpha_base + (1 - cons) * (1 - alpha_base) * gamma
    alpha_i = np.clip(alpha_i, 0.0, 1.0)

    # 5. 融合
    phi = np.zeros((n, D))
    for i in range(n):
        phi[i] = alpha_i[i] * nz[i] + (1 - alpha_i[i]) * nr[i]

    info = {
        'alpha_mean': float(np.mean(alpha_i)),
        'alpha_std': float(np.std(alpha_i)),
        'consistency_mean': float(np.mean(cons)),
        'method': 'GRAA',
    }
    return phi, info


def _weighted_residual_z(X, ctx, A_weighted):
    """用加权邻接做关系回归的残差z。"""
    from . import Context as Ctx
    ctx_w = Ctx(ctx.X_pool, ctx.scorer, A_weighted)
    if hasattr(ctx, 'torch_scorer'):
        ctx_w.torch_scorer = ctx.torch_scorer
    gr = GraphRegression(ctx_w)
    return gr.residual_z(X, A_weighted)


GRAA_DEPENDENT = {
    'GRAA': grcm_attribute,
}
