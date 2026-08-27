# -*- coding: utf-8 -*-
"""图无关归因方法族 (图扰动下 ACR 恒为 1, 改用输入扰动/打分器扰动, v2 必改1)。

  GlobalCF  全局均值轨迹替换 score-drop (来源: 第10篇 two_layer.global_cf_attribute)
  AERec     小 AE 重构误差贡献 (来源: 第10篇 two_layer.ae_recon_attribute)
  Grad      打分器输入梯度×幅值 (来源: 第10篇 two_layer.gradient_attribute;
            本篇打分器为 torch 版 PCA, 解析梯度)
  zDev      稳健偏差读取 (来源: 第10篇 two_layer.zdev_attribute)
  Random    均匀随机对照 (来源: 第10篇 two_layer.random_attribute)

这些方法忽略 ctx.graph; 公平性协议: 其扰动维度为输入噪声/时间切片/打分器替换。
"""
import numpy as np
from . import Context


def global_cf_attribute(X: np.ndarray, ctx: Context) -> np.ndarray:
    """phi_j = s(w) - s(w[j<-全局均值轨迹_j])。"""
    st = ctx.pool_stats
    n, W, D = X.shape
    traj = np.broadcast_to(st['mean'][None, :], (W, D)).astype(np.float32)
    mask = np.arange(D)[None, None, :]
    reps = np.stack([np.where(mask == j, traj, X).astype(np.float32)
                     for j in range(D)], 0)             # (D, n, W, D)
    flat = reps.reshape(D * n, W, D)
    scores = ctx.scorer.score(flat).reshape(D, n)
    s_orig = ctx.scorer.score(X)
    return (s_orig[None, :] - scores).T                 # (n, D)


def ae_rec_attribute(X: np.ndarray, ctx: Context, epochs: int = 30,
                     seed: int = 0) -> np.ndarray:
    """池上训练小 AE; phi_j = 通道 j 的平均重构误差。"""
    import torch
    import torch.nn as nn
    P = ctx.X_pool
    n, W, D = X.shape
    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Flatten(), nn.Linear(W * D, 128), nn.ReLU(),
        nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, W * D))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    Xt = torch.from_numpy(P.reshape(len(P), -1)).float()
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))[:50000]
        loss = nn.functional.mse_loss(net(Xt[perm]), Xt[perm])
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        rec = net(torch.from_numpy(X.reshape(n, -1)).float()).numpy()
    return np.abs(rec.reshape(n, W, D) - X).mean(1)


def grad_attribute(X: np.ndarray, ctx: Context) -> np.ndarray:
    """phi_j = sum_t |ds/dx_tj| * |x_tj| (需要 ctx.torch_scorer 可微打分器)。"""
    import torch
    ts = ctx.torch_scorer
    xt = torch.from_numpy(X.astype(np.float32)).clone()
    xt.requires_grad_(True)
    s = ts.score_t(xt)
    s.sum().backward()
    phi = (xt.grad.abs() * xt.detach().abs()).sum(1).numpy()
    return phi


def zdev_attribute(X: np.ndarray, ctx: Context) -> np.ndarray:
    st = ctx.pool_stats
    return np.abs(X - st['med'][None, None, :]).mean(1) / st['mad'][None, :]


def random_attribute(X: np.ndarray, ctx: Context, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((len(X), X.shape[2]))


GRAPH_FREE = {
    'GlobalCF': global_cf_attribute,
    'AERec': ae_rec_attribute,
    'Grad': grad_attribute,
    'zDev': zdev_attribute,
    'Random': random_attribute,
}


def condattr_attribute(X: np.ndarray, ctx: Context, K: int = 5) -> np.ndarray:
    """CondAttr (reimpl.): GMM 工况识别 + 工况内 K 近邻条件期望模板替换的 score-drop。
    来源: 第10篇 D:/0科研/工作1/第10篇SCI/src/rcca/rcca.py rcca_attribute 的
    归因层简化 (去掉 layer-1 typing / delta); 图无关 (不消费 ctx.graph)。"""
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    P = ctx.X_pool
    n, W, D = X.shape
    t = np.arange(W) - (W - 1) / 2.0

    def feats(Wa):
        mean = Wa.mean(1)
        var = Wa.var(1)
        slope = (Wa * t[None, :, None]).sum(1) / (t ** 2).sum()
        return np.concatenate([mean, slope, var], 1).astype(np.float64)

    scaler = StandardScaler().fit(feats(P))
    pca = PCA(n_components=0.95, random_state=0)
    Fp = pca.fit_transform(scaler.transform(feats(P)))
    gmm = GaussianMixture(8, covariance_type='diag', random_state=0,
                          n_init=2, max_iter=200, reg_covar=1e-3).fit(Fp)
    lab_p = gmm.predict(Fp)
    members = {k: np.where(lab_p == k)[0] for k in range(gmm.n_components)}
    Fq = pca.transform(scaler.transform(feats(X)))
    lab_q = gmm.predict(Fq)
    # 条件期望模板 (工况内 K 近邻均值)
    tmpl = np.empty_like(X)
    for i in range(n):
        mem = members.get(lab_q[i], np.arange(len(P)))
        if len(mem) == 0:
            tmpl[i] = P.mean(0)
            continue
        d = np.linalg.norm(Fp[mem] - Fq[i], axis=1)
        tmpl[i] = P[mem[np.argsort(d)[:K]]].mean(0)
    mask = np.arange(D)[None, None, :]
    reps = np.stack([np.where(mask == j, tmpl, X).astype(np.float32)
                     for j in range(D)], 0)
    scores = ctx.scorer.score(reps.reshape(D * n, W, D)).reshape(D, n)
    s_orig = ctx.scorer.score(X)
    return (s_orig[None, :] - scores).T


GRAPH_FREE['CondAttr'] = condattr_attribute
