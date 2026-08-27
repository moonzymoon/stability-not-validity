# -*- coding: utf-8 -*-
"""GRAA v3: Graph Repair + Attribution — 核心创新.

机制:
  1. 逐边损伤评分: 对每条边, "移除后一致性改善" = 该边的损伤分数
  2. 图修复: 移除高损伤边(可能是伪边)
  3. 在修复图上运行PropRank
  4. 修复后一致性仍低 → 回退zDev

理论保证(Prop 2): 独立边损坏模型下, 损伤信号准确率>1/2时修复单调改善。
"""
import numpy as np
from . import Context, normalize_rows
from .graph_dep import _robust_z, proprank_attribute
from .graph_free import zdev_attribute
from evaluation.metrics import topk
from graphs import graph_ops as go


def _edge_damage_scores(X, ctx, A, M=6):
    """计算每条边的损伤分数: 移除该边后归因一致性的改善量.

    高损伤 = 移除后一致性提高 = 该边在扰乱归因 → 可能是伪边.
    """
    n, W, D = X.shape
    edges = go.edges_from_adj(A)

    # 基线归因(用当前图)
    phi_base = proprank_attribute(X, ctx)
    nz_base = normalize_rows(phi_base)
    base_tops = [topk(nz_base[i], 3) for i in range(n)]

    # 扰动集合(用于评估一致性)
    pert_phis = []
    for mi in range(M):
        rng = np.random.default_rng(42 + mi)
        A_m = go.delete_edges(A, 0.15, rng) if mi % 2 == 0 else \
              go.rewire_edges(A, 0.15, rng)
        ctx_m = ctx.with_graph(A_m)
        pert_phis.append(normalize_rows(proprank_attribute(X, ctx_m)))

    def _consistency_with(phi_nz):
        """归因与扰动集的一致性."""
        vals = []
        for i in range(n):
            base_top = topk(phi_nz[i], 3)
            avg_overlap = np.mean([
                len(base_top & topk(pp[i], 3)) / 3 for pp in pert_phis])
            vals.append(avg_overlap)
        return np.mean(vals)

    base_cons = _consistency_with(nz_base)

    # 逐边损伤评分
    damage = {}
    for (i, j) in edges:
        A_e = A.copy()
        A_e[i, j] = 0.0
        ctx_e = ctx.with_graph(A_e)
        phi_e = proprank_attribute(X, ctx_e)
        nz_e = normalize_rows(phi_e)
        cons_e = _consistency_with(nz_e)
        damage[(i, j)] = cons_e - base_cons  # 正 = 移除后改善

    return damage, base_cons


def graa_v3_attribute(X, ctx, repair_threshold=0.0, fallback_cons=0.5,
                      max_remove_frac=0.3, M=6):
    """GRAA v3: Graph Repair + Attribution.

    Parameters:
    -----------
    X : (n, W, D) anomalous windows
    ctx : Context (must have graph)
    repair_threshold : 移除边的最小损伤分数 (>0 = 移除后一致性必须改善)
    fallback_cons : 修复后一致性低于此值时回退到zDev
    max_remove_frac : 最多移除的边比例
    M : 扰动实例数(一致性评估用)

    Returns:
    --------
    phi : (n, D)
    info : dict (修复边数, 修复前后一致性, 是否回退)
    """
    A = ctx.graph
    D = A.shape[0]
    n = len(X)

    # 1. 逐边损伤评分
    damage, base_cons = _edge_damage_scores(X, ctx, A, M=M)

    # 2. 修复: 移除高损伤边
    edges = sorted(damage.items(), key=lambda kv: -kv[1])
    max_remove = int(len(edges) * max_remove_frac)
    to_remove = [(e, s) for e, s in edges
                 if s > repair_threshold][:max_remove]

    A_repaired = A.copy()
    for (i, j), score in to_remove:
        A_repaired[i, j] = 0.0

    # 3. 在修复图上运行PropRank
    ctx_repaired = ctx.with_graph(A_repaired)
    phi_prop = proprank_attribute(X, ctx_repaired)
    nz_prop = normalize_rows(phi_prop)

    # 4. 评估修复后一致性
    pert_phis_rep = []
    for mi in range(M):
        rng = np.random.default_rng(42 + mi)
        A_m = go.delete_edges(A_repaired, 0.15, rng) if mi % 2 == 0 else \
              go.rewire_edges(A_repaired, 0.15, rng)
        ctx_m = ctx.with_graph(A_m)
        pert_phis_rep.append(normalize_rows(proprank_attribute(X, ctx_m)))

    repaired_cons = np.mean([
        np.mean([len(topk(nz_prop[i], 3) & topk(pp[i], 3)) / 3
                 for pp in pert_phis_rep])
        for i in range(n)])

    # 5. 门控: 修复后一致性足够高 → 用PropRank; 否则 → 回退zDev
    phi_zdev = zdev_attribute(X, ctx)
    nz_zdev = normalize_rows(phi_zdev)

    if repaired_cons >= fallback_cons:
        # 用修复后的PropRank
        w_prop = np.full(n, repaired_cons)
        phi = np.zeros((n, D))
        for i in range(n):
            phi[i] = w_prop[i] * nz_prop[i] + (1 - w_prop[i]) * nz_zdev[i]
        fallback = False
    else:
        # 回退到zDev
        phi = nz_zdev
        fallback = True

    info = {
        'method': 'GRAA-v3',
        'n_edges_original': int(A.sum()),
        'n_edges_removed': len(to_remove),
        'base_consistency': float(base_cons),
        'repaired_consistency': float(repaired_cons),
        'fallback': fallback,
        'removed_edges': [(e, round(s, 4)) for e, s in to_remove[:10]],
    }
    return phi, info
