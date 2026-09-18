# -*- coding: utf-8 -*-
"""AttNN: attention-surrogate 深度归因基线 (修订弹药 A, 供 e65 使用).

思想: 训练一个带变量级加性注意力的 GRU 代理模型模仿打分器 s(·) 在窗口上的
输出, 变量注意力权重 α (对 D softmax) 作为归因 φ。attention-as-explanation
路线的可辩护深度基线。图无关族 (不消费 ctx.graph), 扰动协议与 zDev 等相同
(输入噪声/时间切片, M=8; e1_anchor.INPUT_STRENGTHS)。

结构 (CPU, <0.1M 参数):
  每变量共享 1D-GRU(hidden=16) → 加性注意力 e=v^T tanh(W_h h)
  → 对 W 均值 → 对 D softmax → α (n, D)
  → 时间均值池化 g_j → α 加权求和 → MLP(16→32→1) → ŝ
损失: MSE(ŝ, zscore(s(X_pool))), 训练集=正常池 (部署一致: 打分器也只见池)。
早停 patience=20, max 200 epoch, batch 256, Adam lr=1e-3。

φ 规则 (EXECUTION_PLAN 2.1): 主变体 φ=α; 备选 φ=α*mad_j; 仅当主变体
hit@3<0.15 时按预定规则 (seed0/iforest 上 hit@3 高者) 全局选用, e65 落实。

确定性: seed=20260906 (master), torch.set_num_threads(1), 固定 generator;
同 seed 双跑 α 逐位一致 (e65 验收①)。
"""
import numpy as np
import torch
import torch.nn as nn

MASTER_SEED = 20260906


class AttnSurrogate(nn.Module):
    def __init__(self, hid: int = 16):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hid, batch_first=True)
        self.W_h = nn.Linear(hid, hid, bias=False)
        self.v = nn.Linear(hid, 1, bias=False)
        self.head = nn.Sequential(nn.Linear(hid, 32), nn.ReLU(),
                                  nn.Linear(32, 1))

    def forward(self, x):                      # x: (B, W, D)
        B, W, D = x.shape
        h, _ = self.gru(x.permute(0, 2, 1).reshape(B * D, W, 1))
        e = self.v(torch.tanh(self.W_h(h))).squeeze(-1)   # (B*D, W)
        logits = e.view(B, D, W).mean(-1)                 # (B, D)
        alpha = torch.softmax(logits, dim=-1)
        g = h.view(B, D, W, -1).mean(2)                   # (B, D, hid)
        ctx = (g * alpha.unsqueeze(-1)).sum(1)            # (B, hid)
        s = self.head(ctx).squeeze(-1)                    # (B,)
        return s, alpha


def _forward_alpha(net, X, batch=512):
    net.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).float()
            outs.append(net(xb)[1])
    return torch.cat(outs).numpy().astype(np.float64)


def attnn_alpha(X: np.ndarray, ctx, seed: int = MASTER_SEED,
                max_epochs: int = 150, batch: int = 128,
                lr: float = 1e-3, val_frac: float = 0.1):
    """在**被解释窗口** X 上拟合代理, 返回 (alpha_fn, info)。

    拟合集选择 (v2): 代理解释器惯例是在被解释实例的分布上模仿被解释模型
    (LIME/PGM 式局部保真); 正常池上打分器输出近似白噪声, 无可学结构。
    零标签 — 不接触 R_anom, 公平性协议不变。
    输入按池 median/MAD 归一化 (v3, 诊断 _diag_e65: 原始尺度 GRU 学不动);
    目标为打分器分数的秩变换 (v3, 重尾稳化, 与 top-K 排序用途对齐)。
    固定轮数取终态 (v4): val 集仅 ~15 窗, 早停被 val 噪声误触发
    (4/10 cell 停在第 1 轮), 弃用早停。
    info 含保真字段: val_corr (排序保真), val_mse / const_mse (秩空间)。
    """
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(seed)

    # 输入按池 median/MAD 归一化 (池统计部署可得, 协议一致)
    st = ctx.pool_stats
    med, mad = st['med'], st['mad']

    def norm(A):
        return ((A - med[None, None, :]) / mad[None, None, :]).astype(np.float32)

    F = norm(X)
    s = ctx.scorer.score(X).astype(np.float64)
    from scipy.stats import rankdata
    r = rankdata(s, method='average').astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(F))
    n_val = max(1, int(len(F) * val_frac))
    va, tr = idx[:n_val], idx[n_val:]
    mu, sd = r[tr].mean(), r[tr].std() + 1e-9
    tr_X = torch.from_numpy(F[tr]).float()
    tr_y = torch.from_numpy(((r[tr] - mu) / sd)).float()
    va_X = torch.from_numpy(F[va]).float()
    va_y = ((r[va] - mu) / sd)

    net = AttnSurrogate()
    opt = torch.optim.Adam(net.parameters(), lr)
    for ep in range(max_epochs):
        net.train()
        perm = torch.randperm(len(tr_X), generator=g)
        for i in range(0, len(tr_X), batch):
            sel = perm[i:i + batch]
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(tr_X[sel])[0], tr_y[sel])
            loss.backward()
            opt.step()
    net.eval()
    const_mse = float((va_y ** 2).mean())      # 常数预测(=训练均值)的 val MSE
    with torch.no_grad():
        vp = net(va_X)[0].squeeze(-1).numpy()
        final_mse = float(nn.functional.mse_loss(
            torch.from_numpy(vp), torch.from_numpy(va_y).float()))
    val_corr = float(np.corrcoef(vp, va_y)[0, 1])
    info = dict(val_mse=final_mse, const_mse=const_mse, val_corr=val_corr,
                epochs=max_epochs, seed=seed)
    # alpha_fn 对扰动输入用同一组池统计归一化 (代理不随扰动重训练)
    alpha_fn = lambda Xq: _forward_alpha(net, norm(Xq))   # noqa: E731
    return alpha_fn, info
