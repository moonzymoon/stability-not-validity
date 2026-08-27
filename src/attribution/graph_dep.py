# -*- coding: utf-8 -*-
"""图依赖归因方法族 (图扰动维度的对象, v2 必改1)。

  DPTA-G       图版双路归因: 偏差路(zDev) + 关系路(图约束回归残差), 熵自适应融合
               (融合逻辑沿用第4篇 D:/0科研/工作1/第4篇SCI/src/attribution/type_aware.py
                dual_path_type_aware 的 entropy 融合; 关系路改为显式吃图)
  PropRank     偏差种子 + 图上反向 PageRank 传播 (MonitorRank/CauseInfer/CloudRanger
               一系传播排序方法的代表实现)
  GraphGranger 图约束 lag-1 岭回归摘除归因 (第10篇 Granger 的图约束版,
               参照 D:/0科研/工作1/第10篇SCI/src/evaluation/two_layer.py granger_attribute)

共同点: 归因输出随 ctx.graph 变化 → 图扰动的传导对象。
"""
import numpy as np
from . import Context, normalize_rows


def _robust_z(X: np.ndarray, ctx: Context) -> np.ndarray:
    """偏差路: 每通道窗内对池分布的稳健偏离 (n, D)。"""
    st = ctx.pool_stats
    return np.abs(X - st['med'][None, None, :]).mean(1) / st['mad'][None, :]


def _row_entropy_conf(phi_row: np.ndarray) -> float:
    a = np.abs(phi_row) + 1e-12
    p = a / a.sum()
    ent = -(p * np.log(p)).sum()
    return float(1.0 - ent / np.log(len(phi_row)))


class GraphRegression:
    """对每节点 j 在池上拟合 j ~ pa_G(j) 的线性关系; 查询窗口算残差 z。

    这是关系路的图感知核心: 图错了 → 回归解释子错了 → 残差归因错。
    fit 缓存按 graph 哈希 (同图扰动实例内复用)。
    """

    def __init__(self, ctx: Context):
        self.ctx = ctx
        self._fit = {}

    def _key(self, A):
        return A.tobytes()

    def ensure(self, A: np.ndarray):
        key = self._key(A)
        if key in self._fit:
            return self._fit[key]
        P = self.ctx.X_pool                       # (N, W, D)
        N, W, D = P.shape
        Xf = P.reshape(-1, D)                     # 池内所有时间步
        out = {}
        for j in range(D):
            pa = np.where(A[:, j] > 0)[0]
            if len(pa) == 0:
                out[j] = None                     # 无父: 退化为边际分布
                continue
            Xd = np.hstack([Xf[:, pa], np.ones((len(Xf), 1))])
            lam = 1e-2
            Gm = Xd.T @ Xd + lam * np.eye(Xd.shape[1])
            beta = np.linalg.solve(Gm, Xd.T @ Xf[:, j])
            resid = Xf[:, j] - Xd @ beta
            out[j] = dict(pa=pa, beta=beta,
                          rstd=float(resid.std() + 1e-9))
        self._fit[key] = out
        return out

    def residual_z(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        """查询窗口 (n,W,D) 的关系残差 z (n, D)。"""
        fits = self.ensure(A)
        st = self.ctx.pool_stats
        n, W, D = X.shape
        Xf = X.reshape(-1, D)
        out = np.zeros((n, D))
        for j in range(D):
            f = fits[j]
            if f is None:
                # 无父节点: 关系不可用, 退化为边际 z (由偏差路信息补)
                r = np.abs(Xf[:, j] - st['med'][j]) / st['mad'][j]
                out[:, j] = r.reshape(n, W).mean(1)
                continue
            pa = f['pa']
            Xd = np.hstack([Xf[:, pa], np.ones((len(Xf), 1))])
            resid = np.abs(Xf[:, j] - Xd @ f['beta']) / f['rstd']
            out[:, j] = resid.reshape(n, W).mean(1)
        return out


def dpta_g_attribute(X: np.ndarray, ctx: Context, fusion: str = 'mean'
                     ) -> np.ndarray:
    """DPTA-G 双路融合。fusion='mean' (0.5/0.5, 第4篇实验证明最稳健) 或
    'entropy' (熵自适应, 第4篇 W1 变体; 强偏差下退化为纯偏差路, 作消融)。"""
    A = ctx.graph
    z = _robust_z(X, ctx)
    rel = GraphRegression(ctx).residual_z(X, A)
    nz, nr = normalize_rows(z), normalize_rows(rel)
    if fusion == 'mean':
        return 0.5 * nz + 0.5 * nr
    phi = np.zeros_like(nz)
    for i in range(len(nz)):
        cv, cr = _row_entropy_conf(nz[i]), _row_entropy_conf(nr[i])
        s = cv + cr
        wv, wr = (0.5, 0.5) if s < 1e-8 else (cv / s, cr / s)
        phi[i] = wv * nz[i] + wr * nr[i]
    return phi


def proprank_attribute(X: np.ndarray, ctx: Context, damping: float = 0.5,
                       n_iter: int = 20) -> np.ndarray:
    """PropRank: z 为种子, 沿图反向传播 (下游异常把分数推回上游根因)。"""
    A = ctx.graph
    D = A.shape[0]
    z = _robust_z(X, ctx)
    z = z / (z.max(1, keepdims=True) + 1e-9)
    # 传播矩阵 M[j, i] = w(i->j)/outdeg(i): 节点 j 从其父 i 收集分数
    outdeg = A.sum(1)
    M = np.zeros((D, D))
    for i in range(D):
        if outdeg[i] > 0:
            M[:, i] = A[i, :] / outdeg[i]
    phi = z.copy()
    for _ in range(n_iter):
        phi = (1 - damping) * z + damping * (phi @ M)
    return phi


def graph_granger_attribute(X: np.ndarray, ctx: Context, lag: int = 1
                            ) -> np.ndarray:
    """GraphGranger: 池上拟合 x_i[t] ~ [x_pa(i)[t-1], x_i[t-1]] (仅图允许的边),
    phi_j = 摘除 j 后其子节点预测误差增量之和 (查询窗口上)。"""
    A = ctx.graph
    P = ctx.X_pool
    n, W, D = X.shape
    # 预测子矩阵: 对每个目标 i, predictors = pa(i) ∪ {i}
    preds = {}
    for i in range(D):
        pa = sorted(np.where(A[:, i] > 0)[0].tolist())
        preds[i] = pa + [i] if i not in pa else pa

    def _fit_target(i, drop=None):
        plist = [p for p in preds[i] if p != drop]
        if not plist:
            return None
        Xp = P[:, :-lag][:, :, plist].reshape(-1, len(plist))
        Xp = np.hstack([Xp, np.ones((len(Xp), 1))])
        y = P[:, lag:][:, :, i].reshape(-1)
        lam = 1e-2
        Gm = Xp.T @ Xp + lam * np.eye(Xp.shape[1])
        return dict(plist=plist, beta=np.linalg.solve(Gm, Xp.T @ y))

    def _err(Xq, model, i):
        plist = model['plist']
        Xq_p = Xq[:, :-lag][:, :, plist].reshape(-1, len(plist))
        Xq_p = np.hstack([Xq_p, np.ones((len(Xq_p), 1))])
        y = Xq[:, lag:][:, :, i].reshape(-1)
        return ((Xq_p @ model['beta'] - y) ** 2).reshape(len(Xq), W - lag).sum(1)

    full, drop_models = {}, {}
    for i in range(D):
        full[i] = _fit_target(i)
        for p in preds[i]:
            drop_models[(p, i)] = _fit_target(i, drop=p)

    phi = np.zeros((n, D))
    for i in range(D):
        if full[i] is None:
            continue
        e_full = _err(X, full[i], i)
        for p in preds[i]:
            m = drop_models[(p, i)]
            if m is None:
                continue
            phi[:, p] += (_err(X, m, i) - e_full)
    # phi 可能为负 (摘除反而更好), 保留符号信息; 排序取 argmax 即可
    return phi


GRAPH_DEPENDENT = {
    'DPTA-G': dpta_g_attribute,
    'PropRank': proprank_attribute,
    'GraphGranger': graph_granger_attribute,
}
