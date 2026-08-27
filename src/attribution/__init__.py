# -*- coding: utf-8 -*-
"""归因方法统一上下文与注册表。"""
import numpy as np


class Context:
    """归因方法的共享上下文。

    X_pool: (N,W,D) 正常窗口池
    scorer: 已 fit 的打分器 (score(W)->(n,)); None 表示打分器无关方法
    graph:  (D,D) 邻接矩阵 (图依赖方法用); 图无关方法忽略
    edge_conf: (D,D) 边置信度矩阵 (可选; 加边置信度加权基线用)
    """

    def __init__(self, X_pool, scorer=None, graph=None, edge_conf=None):
        self.X_pool = X_pool
        self.scorer = scorer
        self.graph = graph
        self.edge_conf = edge_conf
        self._pool_stats = None

    @property
    def pool_stats(self):
        """每通道的池 median/MAD (缓存)。"""
        if self._pool_stats is None:
            P = self.X_pool
            self._pool_stats = dict(
                med=np.median(P, axis=(0, 1)),
                mad=np.median(np.abs(P - np.median(P, axis=(0, 1), keepdims=True)),
                              axis=(0, 1)) * 1.4826 + 1e-9,
                mean=P.mean(axis=(0, 1)),
                std=P.std(axis=(0, 1)) + 1e-9,
            )
        return self._pool_stats

    def with_graph(self, graph, edge_conf=None):
        c = Context(self.X_pool, self.scorer, graph, edge_conf)
        if hasattr(self, 'torch_scorer'):
            c.torch_scorer = self.torch_scorer
        return c


def normalize_rows(phi: np.ndarray) -> np.ndarray:
    """行归一化到 [0,1] (每窗口独立), 防止量纲影响融合。"""
    phi = np.abs(phi) + 1e-12
    return phi / phi.sum(1, keepdims=True)


def row_conf(phi: np.ndarray) -> np.ndarray:
    """统一置信度: top-1 份额 (第10篇同款定义)。"""
    p = np.abs(phi) + 1e-12
    return p.max(1) / p.sum(1)
