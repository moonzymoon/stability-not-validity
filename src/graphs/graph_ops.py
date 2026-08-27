# -*- coding: utf-8 -*-
"""图扰动算子库 + 图统计特征 (第11篇基准核心之一)。

三算子 (v2 大纲 §3.1):
  - edge deletion  P_del(e-): 每条边独立以概率 e- 删除
  - edge addition  P_add(e+): 按"不存在边对"的比例 e+ 插入伪边 (保持无环: 只加
    拓扑序前->后的边, 与真值 DAG 的生成方式一致, 避免制造环)
  - rewiring       P_rew(q):  保边数, 随机重连端点

系统性偏差源 (v2 必改2: 解耦现象的主要来源):
  - 换图源: PCMCI 学图(干净) / PCMCI 学图(污染, 含异常段) —— 非随机扰动,
    与随机扰动分开报告。
"""
import numpy as np


def adj_from_edges(n: int, edges) -> np.ndarray:
    A = np.zeros((n, n), dtype=np.float64)
    for (i, j) in edges:
        A[i, j] = 1.0
    return A


def edges_from_adj(A: np.ndarray):
    return [(int(i), int(j)) for i, j in zip(*np.where(A > 0))]


def topological_order(A: np.ndarray, rng: np.random.Generator):
    """给邻接一个 (随机扰动后的) 一致拓扑序: 若 A 无环, 用其约束加边方向。"""
    n = A.shape[0]
    # 简化: 用节点度排序的稳定序 (对无环 A 是有效拓扑序的代理;
    # 实际用 Kahn 算法)
    import networkx as nx
    G = nx.DiGraph((i, j) for i, j in edges_from_adj(A))
    G.add_nodes_from(range(n))
    try:
        return list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        return list(range(n))


def delete_edges(A: np.ndarray, eps: float, rng: np.random.Generator) -> np.ndarray:
    """每条边独立以概率 eps 删除。"""
    if eps <= 0:
        return A.copy()
    mask = rng.random(A.shape) < eps
    out = A.copy()
    out[mask & (A > 0)] = 0.0
    return out


def add_edges(A: np.ndarray, eps: float, rng: np.random.Generator) -> np.ndarray:
    """按不存在边对的比例 eps 加伪边; 仅加拓扑序前->后的边 (保持 DAG)。"""
    n = A.shape[0]
    order = topological_order(A, rng)
    pos = np.empty(n, dtype=int)
    for k, v in enumerate(order):
        pos[v] = k
    absent = [(i, j) for i in range(n) for j in range(n)
              if i != j and A[i, j] == 0 and pos[i] < pos[j]]
    n_add = int(round(eps * len(absent)))
    if n_add <= 0 or not absent:
        return A.copy()
    idx = rng.choice(len(absent), size=min(n_add, len(absent)), replace=False)
    out = A.copy()
    for k in idx:
        i, j = absent[k]
        out[i, j] = 1.0
    return out


def rewire_edges(A: np.ndarray, q: float, rng: np.random.Generator) -> np.ndarray:
    """保边数重连: 随机选 q 比例的边, 删除后随机加回 (拓扑序约束)。"""
    n = A.shape[0]
    edges = edges_from_adj(A)
    if not edges or q <= 0:
        return A.copy()
    k = max(1, int(round(q * len(edges))))
    idx = rng.choice(len(edges), size=min(k, len(edges)), replace=False)
    out = A.copy()
    removed = []
    for i in idx:
        removed.append(edges[i])
        out[edges[i][0], edges[i][1]] = 0.0
    order = topological_order(out, rng)
    pos = np.empty(n, dtype=int)
    for kk, v in enumerate(order):
        pos[v] = kk
    for (i, j) in removed:
        # 尝试保持方向; 若成环则反向
        if pos[i] < pos[j]:
            out[i, j] = 1.0
        else:
            out[j, i] = 1.0
    return out


def edge_recall_precision(A_est: np.ndarray, A_true: np.ndarray):
    """边的召回/精确 (折叠方向: 有向边完全匹配)。"""
    te = set(edges_from_adj(A_true))
    ee = set(edges_from_adj(A_est))
    if not te:
        return 1.0, 1.0
    rec = len(te & ee) / len(te)
    prec = (len(te & ee) / len(ee)) if ee else 1.0
    return rec, prec


def edge_f1(A_est, A_true):
    r, p = edge_recall_precision(A_est, A_true)
    return 0.0 if r + p == 0 else 2 * r * p / (r + p)


# ---------------- 可观测图统计特征 (可信度预测器特征工程) ----------------
def graph_features(A: np.ndarray) -> dict:
    """部署时可观测的图统计量 (不依赖真值图)。"""
    n = A.shape[0]
    E = A.sum()
    dens = E / (n * (n - 1)) if n > 1 else 0.0
    indeg = A.sum(0)
    outdeg = A.sum(1)
    def _ent(x):
        s = x.sum()
        if s <= 0:
            return 0.0
        p = x / s
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())
    return dict(
        n_nodes=n, n_edges=float(E), density=float(dens),
        indeg_entropy=_ent(indeg), outdeg_entropy=_ent(outdeg),
        indeg_max=float(indeg.max()), outdeg_max=float(outdeg.max()),
        indeg_mean=float(indeg.mean()), indeg_std=float(indeg.std()),
        sink_frac=float((indeg == 0).mean()), source_frac=float((outdeg == 0).mean()),
    )


PERTURB_FAMILIES = ('del', 'add', 'rew')
