# -*- coding: utf-8 -*-
"""E13 (方案7/8a): 训练版 GNN 排序器 + AE 打分器进 SCM 主网格."""
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
from attribution import Context
from attribution.graph_dep import _robust_z
from attribution.graph_free import GRAPH_FREE
from evaluation import metrics as M
from scripts.e1_anchor import perturb_graph, stable_seed, M_PERTURB

CACHE = os.path.join(SRC, '_cache')


# ================= 方案7: 训练版 GNN 排序器 =================
class GNNRanker:
    """2 层图卷积 + 线性读出的根因排序器 (torch, 在合成训练集上学参数).
    输入特征: 稳健偏差 z (1维/节点); 图: 邻接(行归一化+自环, 反向传播).
    训练: BCE on root-cause indicator (合成注入真值)."""

    def __init__(self, d=15, hidden=16, seed=0):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        self.torch = torch
        self.net = nn.ModuleList([
            nn.Linear(2 * d, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, d)])
        self.d = d

    def _propagate(self, A):
        t = self.torch
        Ahat = (A + np.eye(self.d)).astype(np.float32)
        Ahat /= np.maximum(Ahat.sum(1, keepdims=True), 1)
        return t.tensor(Ahat.T.copy())          # 反向传播 (下游->上游回流)

    def fit(self, samples, graphs, epochs=8, lr=5e-3):
        """samples: list of (z(n,d), y(n,d)); graphs: list of adj per sample."""
        t = self.torch
        params = [p for m in self.net for p in m.parameters()]
        opt = t.optim.Adam(params, lr=lr)
        Ahs = [self._propagate(A) for A in graphs]
        rng2 = np.random.default_rng(0)
        for ep in range(epochs):
            tot = 0.0
            sel = rng2.choice(len(samples), size=min(400, len(samples)),
                              replace=False)
            for si in sel:
                (z, y), Ah = samples[si], Ahs[si]
                zx = t.tensor(z.astype(np.float32))
                H = t.cat([zx @ Ah, zx], 1)          # (n, d+d) 传播后+原始
                for m in self.net:
                    H = m(H)
                yl = t.tensor(y.astype(np.float32))
                pos = (yl > 0.5).sum()
                neg = (yl <= 0.5).sum()
                w = t.where(yl > 0.5, t.tensor(neg / max(pos, 1.0)),
                            t.tensor(1.0))
                loss = t.nn.functional.binary_cross_entropy_with_logits(
                    H, yl, weight=w)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss)
            if ep % 50 == 0:
                print(f'    epoch {ep} loss {tot / len(samples):.4f}', flush=True)
        return self

    def rank(self, z, A):
        t = self.torch
        with t.no_grad():
            zx = t.tensor(z[None].astype(np.float32))
            Ah = self._propagate(A)
            H = t.cat([zx @ Ah, zx], 1)
            for m in self.net:
                H = m(H)
            return H[0].numpy()


def p7_gnn():
    import torch  # noqa
    records = []
    # 训练集: seed 0-2 (偏差型中档), 测试: seed 3-4 (含弱/强档泛化)
    train_s, train_g = [], []
    for seed in range(3):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2, seed=seed,
                        noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        scorer = make_scorer('iforest').fit(w['X_pool'])
        ctx = Context(w['X_pool'], scorer=scorer)
        z = _robust_z(w['X_anom'], ctx)
        A = s['adjacency'].astype(float)
        n = len(z)
        # 每 episode 一条训练样本 (窗口平均 z, 根因指示)
        # 窗级样本 (每窗口一条) + 图扰动增广 (每窗 2 个扰动图)
        rng = np.random.default_rng(seed)
        for i in range(len(z)):
            ye = np.zeros((1, 15))
            ye[0, list(w['R_anom'][i])] = 1.0
            train_s.append((z[i][None], ye))
            train_g.append(A)
            for _ in range(2):
                A_p = go.rewire_edges(A, 0.2, rng)
                train_s.append((z[i][None], ye))
                train_g.append(A_p)
    gnn = GNNRanker(seed=0)
    gnn.fit(train_s, train_g)
    # 测试: seed 3/4 三档难度 × 3 图源
    for seed in (3, 4):
        for mag in (0.25, 0.5, 1.5):
            s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                            seed=seed, noise_std=0.5, magnitude=mag)
            w = build_windows(s)
            A_true = s['adjacency'].astype(float)
            A_pcmci = pcmci_graph(s['normal'])['adj']
            scorer = make_scorer('iforest').fit(w['X_pool'])
            ctx = Context(w['X_pool'], scorer=scorer)
            z = _robust_z(w['X_anom'], ctx)
            for gname, A in (('true', A_true), ('pcmci', A_pcmci)):
                phi = np.stack([gnn.rank(zi, A) for zi in z])
                records.append(dict(exp='p7', dataset=f'scm{seed}_m{mag}',
                                    graph_source=gname,
                                    hit3=float(M.mean_hit(phi, w['R_anom'], 3))))
    json.dump(records, open(os.path.join(CACHE, 'e13_p7.json'), 'w'),
              default=float)
    agg = collections.defaultdict(list)
    for r in records:
        agg[(r['dataset'].split('_m')[1], r['graph_source'])].append(r['hit3'])
    print('=== P7 trained GNN ranker (mag, source): hit@3 ===')
    for k in sorted(agg, key=str):
        print(f'  {str(k):18s} hit={np.mean(agg[k]):.3f}')


# ================= 方案8a: AE 打分器进 SCM 主网格 =================
def p8a_ae_grid():
    from scripts.e8_round6 import AEScorer, gcn_rank_attribute
    from attribution.graph_dep import GRAPH_DEPENDENT
    records = []
    for seed in range(5):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2, seed=seed,
                        noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        A_true = s['adjacency'].astype(float)
        A_anom = pcmci_graph(s['series'])['adj']
        scorer = AEScorer(seed=seed).fit(w['X_pool'])
        ctx0 = Context(w['X_pool'], scorer=scorer)
        for mname, fn in list(GRAPH_DEPENDENT.items()) + [
                ('GCN-Rank', gcn_rank_attribute)]:
            for gname, A in (('true', A_true), ('pcmci_anom', A_anom)):
                phi = fn(w['X_anom'], ctx0.with_graph(A))
                acrs = []
                for mi in range(M_PERTURB):
                    rng = np.random.default_rng(stable_seed(
                        'p8a', seed, mname, gname, mi))
                    A_m = perturb_graph(A, ['del', 'add', 'rew'][mi % 3],
                                        0.2, rng)
                    phi_m = fn(w['X_anom'], ctx0.with_graph(A_m))
                    acrs.append(M.acr_at_k(phi_m, phi, 3).mean())
                records.append(dict(
                    exp='p8a', dataset=f'scm{seed}', scorer='ae',
                    method=mname, graph_source=gname,
                    hit3=float(M.mean_hit(phi, w['R_anom'], 3)),
                    acr3=float(np.mean(acrs))))
        for mname in ('zDev', 'AERec', 'GlobalCF'):
            fn = GRAPH_FREE[mname]
            phi = fn(w['X_anom'], ctx0)
            records.append(dict(exp='p8a', dataset=f'scm{seed}', scorer='ae',
                                method=mname, graph_source='none',
                                hit3=float(M.mean_hit(phi, w['R_anom'], 3)),
                                acr3=None))
        print(f'[P8a] scm{seed} done', flush=True)
    json.dump(records, open(os.path.join(CACHE, 'e13_p8a.json'), 'w'),
              default=float)
    agg = collections.defaultdict(lambda: dict(h=[], a=[]))
    for r in records:
        agg[(r['method'], r['graph_source'])]['h'].append(r['hit3'])
        if r.get('acr3') is not None:
            agg[(r['method'], r['graph_source'])]['a'].append(r['acr3'])
    print('=== P8a AE-scorer main grid (method, source): hit3/acr3 ===')
    for k in sorted(agg, key=str):
        h = np.mean(agg[k]['h'])
        a = np.mean(agg[k]['a']) if agg[k]['a'] else float('nan')
        print(f'  {str(k):38s} hit={h:.3f} acr={a:.3f}')


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', 'p7'):
        p7_gnn()
    if which in ('all', 'p8a'):
        p8a_ae_grid()
