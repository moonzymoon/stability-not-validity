# -*- coding: utf-8 -*-
"""打分器统一接口 (三族经典 + 可微 PCA)。

来源: 接口设计参照第10篇 D:/0科研/工作1/第10篇SCI/src/scorers_rescore.py 的
scorer 抽象, 本篇在合成数据上直接自实现 (iforest/PCA/OCSVM 三族, 与第10篇
五检测器中的三经典族同型)。打分器质量分层 (必改: 隔离"窗口选错"与"归因错")
通过 train_frac 参数控制训练数据量实现, 并输出打分器自身的检测 AUROC 作为
"质量"操作化度量。
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM


def _flat(W: np.ndarray) -> np.ndarray:
    return W.reshape(len(W), -1).astype(np.float64)


class IForestScorer:
    name = 'iforest'
    family = 'tree-ensemble'

    def __init__(self, train_frac=1.0, seed=0):
        self.train_frac = train_frac
        self.m = IsolationForest(n_estimators=200, random_state=seed,
                                 contamination='auto')

    def fit(self, pool):
        F = _flat(pool)
        if self.train_frac < 1.0:
            rng = np.random.default_rng(7)
            idx = rng.choice(len(F), size=max(50, int(len(F) * self.train_frac)),
                             replace=False)
            F = F[idx]
        self.m.fit(F)
        return self

    def score(self, W):
        # decision_function: 越大越正常 → 取负使越大越异常
        return -self.m.decision_function(_flat(W))


class PCAScorer:
    name = 'pca'
    family = 'smooth-linear'

    def __init__(self, train_frac=1.0, seed=0, var_keep=0.95):
        self.train_frac = train_frac
        self.pca = PCA(n_components=var_keep, random_state=seed)

    def fit(self, pool):
        F = _flat(pool)
        if self.train_frac < 1.0:
            rng = np.random.default_rng(7)
            idx = rng.choice(len(F), size=max(50, int(len(F) * self.train_frac)),
                             replace=False)
            F = F[idx]
        self.pca.fit(F)
        self.mu_ = self.pca.mean_
        self.comp_ = self.pca.components_          # (k, p)
        return self

    def score(self, W):
        F = _flat(W)
        rec = self.pca.inverse_transform(self.pca.transform(F))
        return ((F - rec) ** 2).sum(1)


class OCSVMScorer:
    name = 'ocsvm'
    family = 'kernel-smooth'

    def __init__(self, train_frac=1.0, seed=0, nu=0.05):
        self.train_frac = train_frac
        self.m = OneClassSVM(kernel='rbf', nu=nu)

    def fit(self, pool):
        F = _flat(pool)
        if self.train_frac < 1.0:
            rng = np.random.default_rng(7)
            idx = rng.choice(len(F), size=max(50, int(len(F) * self.train_frac)),
                             replace=False)
            F = F[idx]
        self.m.fit(F)
        return self

    def score(self, W):
        return -self.m.decision_function(_flat(W))


def detection_auroc(scorer, X_pool, X_anom):
    """打分器质量操作化度量: 正常池 vs 异常窗口的检测 AUROC。"""
    from sklearn.metrics import roc_auc_score
    s_n = scorer.score(X_pool)
    s_a = scorer.score(X_anom)
    y = np.r_[np.zeros(len(s_n)), np.ones(len(s_a))]
    s = np.r_[s_n, s_a]
    return float(roc_auc_score(y, s))


def make_scorer(kind: str, train_frac=1.0, seed=0):
    if kind == 'iforest':
        return IForestScorer(train_frac, seed)
    if kind == 'pca':
        return PCAScorer(train_frac, seed)
    if kind == 'ocsvm':
        return OCSVMScorer(train_frac, seed)
    raise ValueError(kind)


# ---------------- 可微打分器 (Grad 方法用; torch 实现的 PCA 重构误差) ----------------
class TorchPCAScorer:
    """与 PCAScorer 同型的 torch 版, 支持解析梯度 (Grad 归因的前提)。"""
    name = 'pca-torch'
    family = 'smooth-linear'

    def __init__(self, pca: PCAScorer):
        import torch
        self.torch = torch
        self.comp = torch.tensor(pca.comp_, dtype=torch.float32)   # (k,p)
        self.mu = torch.tensor(pca.mu_, dtype=torch.float32)       # (p,)

    def score_t(self, W):
        """W: torch (n, W, D) requires_grad → score (n,)。"""
        t = self.torch
        F = W.reshape(len(W), -1)
        c = F - self.mu
        z = c @ self.comp.T
        rec = z @ self.comp + self.mu
        return ((F - rec) ** 2).sum(1)
