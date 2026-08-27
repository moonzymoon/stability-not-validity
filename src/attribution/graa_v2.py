# -*- coding: utf-8 -*-
"""GRAA v2: Graph-Reliability-Aware Attribution — 基于PropRank的自适应版本.

核心改进(v2): 
  - 基础方法改为PropRank(真传导方法,对图质量敏感)
  - 一致性低时切换到zDev(图无关后备)
  - 边置信度加权传播(高置信边优先传播)
  - 这样GRAA在好图上≈PropRank,在坏图上退化为zDev(而非崩溃)
"""
import numpy as np
from . import Context, normalize_rows
from .graph_dep import _robust_z, proprank_attribute
from .graph_free import zdev_attribute
from evaluation.metrics import topk


def graa_v2_attribute(X, ctx, edge_conf=None, gamma=0.7, M=4):
    """GRAA v2: PropRank base + consistency-gated fallback to zDev.

    Parameters:
    -----------
    X : (n, W, D) anomalous windows
    ctx : Context (must have graph set)
    edge_conf : (D, D) edge confidence (optional)
    gamma : gating strength (0=no gating, 1=hard switch)
    M : number of perturbation instances for consistency estimate

    Returns:
    --------
    phi : (n, D)
    info : dict
    """
    A = ctx.graph
    n, W, D = X.shape
    from graphs import graph_ops as go

    # 1. PropRank on the given graph
    phi_prop = proprank_attribute(X, ctx)

    # 2. zDev (graph-free fallback)
    phi_zdev = zdev_attribute(X, ctx)
    nz_prop = normalize_rows(phi_prop)
    nz_zdev = normalize_rows(phi_zdev)

    # 3. Edge-confidence weighted graph for propagation
    if edge_conf is not None:
        A_weighted = A * edge_conf
        ctx_weighted = ctx.with_graph(A_weighted)
        phi_prop_w = proprank_attribute(X, ctx_weighted)
        nz_prop_w = normalize_rows(phi_prop_w)
    else:
        nz_prop_w = nz_prop

    # 4. Per-window consistency (perturbation-based)
    cons = np.zeros(n)
    for i in range(n):
        base_top = topk(nz_prop_w[i], 3)
        overlaps = []
        for mi in range(M):
            rng = np.random.default_rng(42 + mi)
            A_m = go.delete_edges(A, 0.2, rng) if mi % 2 == 0 else \
                  go.rewire_edges(A, 0.2, rng)
            ctx_m = ctx.with_graph(A_m)
            phi_m = proprank_attribute(X[i:i+1], ctx_m)
            nz_m = normalize_rows(phi_m)
            pert_top = topk(nz_m[0], 3)
            overlaps.append(len(base_top & pert_top) / 3)
        cons[i] = np.mean(overlaps)

    # 5. Adaptive weighting: high consistency → PropRank, low → zDev
    # w_prop = cons^gamma (gamma controls sharpness)
    w_prop = np.power(cons, gamma)
    w_zdev = 1.0 - w_prop

    # 6. Fuse
    phi = np.zeros((n, D))
    for i in range(n):
        phi[i] = w_prop[i] * nz_prop_w[i] + w_zdev[i] * nz_zdev[i]

    info = {
        'method': 'GRAA-v2',
        'w_prop_mean': float(np.mean(w_prop)),
        'w_prop_std': float(np.std(w_prop)),
        'consistency_mean': float(np.mean(cons)),
        'gamma': gamma,
    }
    return phi, info
