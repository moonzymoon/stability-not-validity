# -*- coding: utf-8 -*-
"""图源: 真值图 / PCMCI 学图(干净) / PCMCI 学图(污染)。

PCMCI wrapper 来源: 复制自第8篇 D:/0科研/工作1/第8篇SCI/src/causal/pcmci_discover.py
(tigramite 5.2.10.1 已验证用法: DataFrame 不传 data_type; graph[i,j,tau]='-->'
表示 i→j; fdr_bh 纠正)。

系统性偏差源设计 (v2 必改2): 实践中的图常从**含异常**的数据学得 —— 污染学图
相对干净学图的偏差是"系统性"而非"随机"的, 是解耦现象的主预期来源。
"""
import numpy as np


def pcmci_graph(data: np.ndarray, tau_max: int = 5, pc_alpha: float = 0.05,
                seed: int = 0) -> dict:
    """对 (T, N) 数据跑 PCMCI, 返回 dict(adj (N,N), edges set, f1 所需原料)。"""
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr

    N = data.shape[1]
    df = pp.DataFrame(data, var_names=[f'x{i}' for i in range(N)])
    pcmci = PCMCI(dataframe=df, cond_ind_test=ParCorr(), verbosity=0)
    res = pcmci.run_pcmci(tau_min=1, tau_max=tau_max, pc_alpha=pc_alpha,
                          fdr_method='fdr_bh')
    graph = res['graph']
    edges = set()
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            for tau in range(graph.shape[2]):
                if tau < 1:
                    continue
                if str(graph[i, j, tau]).strip() in ('-->', 'o->', 'x->'):
                    edges.add((i, j))
    A = np.zeros((N, N))
    for (i, j) in edges:
        A[i, j] = 1.0
    return dict(adj=A, edges=edges, val_matrix=res['val_matrix'])


def pcmci_edge_conf(data: np.ndarray, tau_max: int = 5,
                    pc_alpha: float = 0.05) -> np.ndarray:
    """边置信度矩阵 |val_matrix| 最大值 (加边置信度加权基线 / 预测器特征用)。"""
    r = pcmci_graph(data, tau_max, pc_alpha)
    return np.abs(r['val_matrix']).max(2) * r['adj']


def graph_sources(sample: dict, tau_max: int = 5) -> dict:
    """三图源: true / pcmci_clean(正常段) / pcmci_anom(含异常整段)。

    返回 {name: adj (N,N)}。
    """
    from graphs.graph_ops import adj_from_edges, edge_recall_precision

    A_true = sample['adjacency'].astype(float)
    A_clean = pcmci_graph(sample['normal'], tau_max=tau_max)['adj']
    A_anom = pcmci_graph(sample['series'], tau_max=tau_max)['adj']
    out = {'true': A_true, 'pcmci_clean': A_clean, 'pcmci_anom': A_anom}
    for name, A in out.items():
        r, p = edge_recall_precision(A, A_true)
        out[name] = A
        print(f"    [graph:{name}] edges={int(A.sum())} recall={r:.2f} prec={p:.2f}",
              flush=True)
    return out
