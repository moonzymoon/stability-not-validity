# -*- coding: utf-8 -*-
"""E8: Round-6 外部评审补强实验.

A  GCN-Rank: 2 层 GCN 式传播排序 (现代图依赖代表, 诚实标注为 simplified)
B  AE scorer: 深度可微打分器 (第 4 打分器族) + Grad on AE + 主配置
C  corr-graph: 相关性阈值图 (非 PCMCI 图源, 实践常用的系统偏差源)
D  DPTA-G 输入扰动 ACR (混合方法的公平性补测)
E  窗口长度消融 W=8/16/32
输出: _cache/e8_gcnrank.json / e8_aescorer.json / e8_corrgraph.json /
      e8_dpta_input.json / e8_windowsens.json
"""
import os
import sys
import json
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.scm import make_sample, build_windows
from graphs.sources import pcmci_graph
from graphs import graph_ops as go
from scorers import make_scorer
from attribution import Context, normalize_rows
from attribution.graph_dep import (_robust_z, GRAPH_DEPENDENT,
                                   dpta_g_attribute)
from attribution.graph_free import (GRAPH_FREE, zdev_attribute,
                                    ae_rec_attribute)
from evaluation import metrics as M
from scripts.e1_anchor import perturb_graph, perturb_input, stable_seed, M_PERTURB

CACHE = os.path.join(SRC, '_cache')


# ---------------------------------------------------------------- A: GCN-Rank
def gcn_rank_attribute(X, ctx, n_layers=2, alpha_self=1.0):
    """2 层 GCN 式传播排序: H = ReLU(Ahat @ H), H0 = 归一化偏差种子.
    Ahat = A + I (自环), 行归一化 —— GCN 传播的简化实现 (无参数训练,
    Kipf-Welling 谱传播的线性核), 作为'现代图依赖代表'的诚实简化."""
    A = ctx.graph
    D = A.shape[0]
    z = _robust_z(X, ctx)
    z = z / (z.max(1, keepdims=True) + 1e-9)
    # 根因排序用反向传播: 分数沿因果边逆向回流 (下游高分推给上游根因,
    # GNN-for-RCA 文献在 reversed graph 上做 random walk 的同一惯例)
    Ahat = A.T + alpha_self * np.eye(D)
    deg = Ahat.sum(1, keepdims=True)
    Ahat = Ahat / np.maximum(deg, 1)
    H = z
    for _ in range(n_layers):
        H = np.tanh(H @ Ahat)
    return H


# ---------------------------------------------------------------- B: AE scorer
class AEScorer:
    """深度可微打分器: 池上训练的小 AE, 分数=窗口重构误差 (torch 可微)."""
    name = 'ae'
    family = 'deep'

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, pool):
        import torch
        import torch.nn as nn
        P = pool
        torch.manual_seed(self.seed)
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(P.shape[1] * P.shape[2], 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, P.shape[1] * P.shape[2]))
        opt = torch.optim.Adam(self.net.parameters(), 1e-3)
        Xt = torch.from_numpy(P.reshape(len(P), -1)).float()
        for ep in range(30):
            perm = torch.randperm(len(Xt))[:50000]
            loss = nn.functional.mse_loss(self.net(Xt[perm]), Xt[perm])
            opt.zero_grad(); loss.backward(); opt.step()
        self.net.eval()
        return self

    def score(self, W):
        import torch
        with torch.no_grad():
            t = torch.from_numpy(W.reshape(len(W), -1)).float()
            rec = self.net(t).numpy().reshape(W.shape)
        return ((W - rec) ** 2).sum((1, 2))

    def score_t(self, W):
        import torch
        t = W.reshape(len(W), -1)
        rec = self.net(t)
        return ((t - rec) ** 2).sum(1)

    def grad_phi(self, W):
        import torch
        xt = torch.from_numpy(W.astype(np.float32)).clone()
        xt.requires_grad_(True)
        s = self.score_t(xt)
        s.sum().backward()
        return (xt.grad.abs() * xt.detach().abs()).sum(1).numpy()


def corr_graph(data, thresh=0.3, max_lag=3):
    """相关性阈值图: 通道间最大滞后 |Pearson| > thresh 连边, 方向=滞后领先者.
    实践常用 (无因果发现), 系统偏差模式与 PCMCI 不同 (无方向控制+阈值敏感)."""
    T, D = data.shape
    A = np.zeros((D, D))
    xc = (data - data.mean(0)) / (data.std(0) + 1e-9)
    for i in range(D):
        for j in range(D):
            if i == j:
                continue
            best, best_lag = 0.0, 0
            for lag in range(0, max_lag + 1):
                if lag == 0:
                    r = abs(np.corrcoef(xc[:, i], xc[:, j])[0, 1])
                else:
                    r = abs(np.corrcoef(xc[lag:, i], xc[:-lag, j])[0, 1])
                if r > best:
                    best, best_lag = r, lag
            if best > thresh:
                # 方向: 若 i 领先 j (lag>0 时 i 的过去预测 j 的现在) → i→j
                if best_lag > 0:
                    A[i, j] = 1
                else:
                    A[max(i, j), min(i, j)] = 1   # 同步: 约定高索引→低索引
    return A


def run_ABC():
    """A(GCN-Rank)+B(AE scorer)+C(corr-graph) 的主配置网格."""
    records = []
    for seed in range(5):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2, seed=seed,
                        noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        A_true = s['adjacency'].astype(float)
        A_pcmci = pcmci_graph(s['series'])['adj']
        A_corr = corr_graph(s['normal'])
        r_true, _ = go.edge_recall_precision(A_corr, A_true)
        graphs = {'true': A_true, 'pcmci_anom': A_pcmci, 'corr': A_corr}
        for sk in ('iforest', 'pca', 'ae'):
            if sk == 'ae':
                scorer = AEScorer(seed=seed).fit(w['X_pool'])
            else:
                scorer = make_scorer(sk).fit(w['X_pool'])
            ctx0 = Context(w['X_pool'], scorer=scorer)
            for mname, fn, gdep in (
                    ('GCN-Rank', gcn_rank_attribute, True),
                    ('PropRank', GRAPH_DEPENDENT['PropRank'], True)):
                for gname, A in graphs.items():
                    ctx = ctx0.with_graph(A)
                    phi = fn(w['X_anom'], ctx)
                    hits = dict(
                        hit1=float(M.mean_hit(phi, w['R_anom'], 1)),
                        hit3=float(M.mean_hit(phi, w['R_anom'], 3)))
                    acrs = []
                    for mi in range(M_PERTURB):
                        rng = np.random.default_rng(stable_seed(
                            'e8', seed, sk, mname, gname, mi))
                        A_m = perturb_graph(A, ['del', 'add', 'rew'][mi % 3],
                                            0.2, rng)
                        phi_m = fn(w['X_anom'], ctx0.with_graph(A_m))
                        acrs.append(M.acr_at_k(phi_m, phi, 3).mean())
                    records.append(dict(
                        exp='e8_grid', dataset=f'scm{seed}', scorer=sk,
                        method=mname, graph_source=gname, **hits,
                        acr3=float(np.mean(acrs))))
            # Grad on AE scorer (深度可微打分器上的梯度归因)
            if sk == 'ae':
                phi = scorer.grad_phi(w['X_anom'])
                records.append(dict(
                    exp='e8_grid', dataset=f'scm{seed}', scorer='ae',
                    method='Grad-AE', graph_source='none',
                    hit1=float(M.mean_hit(phi, w['R_anom'], 1)),
                    hit3=float(M.mean_hit(phi, w['R_anom'], 3)),
                    acr3=None))
        print(f'[e8] scm{seed} done (corr-graph recall={r_true:.2f})',
              flush=True)
    json.dump(records, open(os.path.join(CACHE, 'e8_grid.json'), 'w'),
              default=float)
    agg = collections.defaultdict(list)
    for r in records:
        agg[(r['method'], r['scorer'], r['graph_source'])].append(
            (r['hit3'], r.get('acr3')))
    print('=== e8 grid (method, scorer, source): hit3 / acr3 ===')
    for k in sorted(agg, key=str):
        h = np.mean([x[0] for x in agg[k]])
        a = [x[1] for x in agg[k] if x[1] is not None]
        print(f'  {str(k):48s} hit={h:.3f} acr={np.mean(a):.3f}' if a
              else f'  {str(k):48s} hit={h:.3f} acr=--')
    return records


def run_D():
    """DPTA-G 输入扰动 ACR (混合方法公平性)."""
    records = []
    for seed in range(5):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2, seed=seed,
                        noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        A = s['adjacency'].astype(float)
        scorer = make_scorer('iforest').fit(w['X_pool'])
        ctx0 = Context(w['X_pool'], scorer=scorer)
        for mname, fn in (('DPTA-G', lambda X, c: dpta_g_attribute(X, c)),
                          ('zDev', zdev_attribute),
                          ('AERec', ae_rec_attribute)):
            phi0 = fn(w['X_anom'], ctx0.with_graph(A) if mname == 'DPTA-G'
                      else ctx0)
            for fam, st in (('noise', 0.2), ('noise', 0.5), ('slice', 1)):
                acrs = []
                for mi in range(4):
                    rng = np.random.default_rng(stable_seed(
                        'e8d', seed, mname, fam, st, mi))
                    X_m = perturb_input(w['X_anom'], fam, st, rng, ctx0)
                    ctx_m = ctx0.with_graph(A) if mname == 'DPTA-G' else ctx0
                    phi_m = fn(X_m, ctx_m)
                    acrs.append(M.acr_at_k(phi_m, phi0, 3).mean())
                records.append(dict(exp='e8_dpta_input', dataset=f'scm{seed}',
                                    method=mname, family=fam, strength=st,
                                    acr3=float(np.mean(acrs))))
    json.dump(records, open(os.path.join(CACHE, 'e8_dpta_input.json'), 'w'),
              default=float)
    agg = collections.defaultdict(list)
    for r in records:
        agg[(r['method'], r['family'], r['strength'])].append(r['acr3'])
    print('=== D: input-perturbation ACR (method, fam, st) ===')
    for k in sorted(agg, key=str):
        print(f'  {str(k):40s} acr={np.mean(agg[k]):.3f}')


def run_E():
    """窗口长度消融."""
    records = []
    for seed in range(3):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2, seed=seed,
                        noise_std=0.5, magnitude=0.5)
        A = s['adjacency'].astype(float)
        for W in (8, 16, 32):
            w = build_windows(s, window=W)
            scorer = make_scorer('iforest').fit(w['X_pool'])
            ctx0 = Context(w['X_pool'], scorer=scorer)
            for mname, fn, gdep in (
                    ('PropRank', GRAPH_DEPENDENT['PropRank'], True),
                    ('GCN-Rank', gcn_rank_attribute, True),
                    ('DPTA-G', dpta_g_attribute, True),
                    ('zDev', zdev_attribute, False),
                    ('AERec', ae_rec_attribute, False)):
                ctx = ctx0.with_graph(A) if gdep else ctx0
                phi = fn(w['X_anom'], ctx)
                h = float(M.mean_hit(phi, w['R_anom'], 3))
                records.append(dict(exp='e8_winsens', dataset=f'scm{seed}',
                                    window=W, method=mname, hit3=h))
    json.dump(records, open(os.path.join(CACHE, 'e8_windowsens.json'), 'w'),
              default=float)
    agg = collections.defaultdict(list)
    for r in records:
        agg[(r['method'], r['window'])].append(r['hit3'])
    print('=== E: window sensitivity (method, W): hit3 ===')
    for k in sorted(agg, key=str):
        print(f'  {str(k):32s} hit={np.mean(agg[k]):.3f}')


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', 'abc'):
        run_ABC()
    if which in ('all', 'd'):
        run_D()
    if which in ('all', 'e'):
        run_E()
