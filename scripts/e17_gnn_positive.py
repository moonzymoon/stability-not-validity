# -*- coding: utf-8 -*-
"""E17: GNN正结果 — 同分布训练测试(SCM seed内LOMO) vs 跨分布(LODO).
关键: 之前失败因为跨seed迁移; 现在测试same-protocol内的表现."""
import os, sys, json, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.scm import make_sample, build_windows
from graphs.sources import pcmci_graph
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import _robust_z
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')


class SimpleGNNRanker:
    """2层图卷积排序器 — episode级训练, episode级评估(同seed内)."""
    def __init__(self, d=15, hidden=32, seed=0):
        import torch, torch.nn as nn
        torch.manual_seed(seed)
        self.torch = torch
        self.d = d
        # 反向传播矩阵(预计算)
        self.net = nn.Sequential(
            nn.Linear(2*d, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d))

    def _feat(self, z, A):
        Ahat = (A + np.eye(self.d)).astype(np.float32)
        Ahat /= np.maximum(Ahat.sum(1, keepdims=True), 1)
        # 传播后的z + 原始z
        return np.hstack([z @ Ahat.T, z]).astype(np.float32)

    def fit_episode(self, episodes, A, epochs=100, lr=1e-3):
        """episodes: list of (z_avg (d,), root_set set)"""
        import torch
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        X = torch.tensor(np.stack([self._feat(z, A) for z, _ in episodes]))
        Y = torch.zeros(len(episodes), self.d)
        for i, (_, roots) in enumerate(episodes):
            for r in roots:
                Y[i, r] = 1.0
        for ep in range(epochs):
            out = self.net(X)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                out, Y, weight=torch.where(Y > 0.5,
                    torch.tensor(len(Y.flatten()) / max((Y > 0.5).sum(), 1), dtype=torch.float32),
                    torch.tensor(1.0)))
            opt.zero_grad(); loss.backward(); opt.step()
        return self

    def rank(self, z, A):
        import torch
        with torch.no_grad():
            x = torch.tensor(self._feat(z, A)[None])
            return self.net(x)[0].numpy()


def run():
    records = []
    for seed in range(5):
        for mag in (0.5,):  # 中档
            s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                            seed=seed, noise_std=0.5, magnitude=mag)
            w = build_windows(s)
            A_true = s['adjacency'].astype(float)
            A_pcmci = pcmci_graph(s['normal'])['adj']
            scorer = make_scorer('iforest').fit(w['X_pool'])
            ctx = Context(w['X_pool'], scorer=scorer)
            z_all = _robust_z(w['X_anom'], ctx)

            # 构造episode级数据: 按anomaly_idx分组
            ep_data = collections.defaultdict(list)
            for i, meta in enumerate(w['meta_anom']):
                ep_data[meta['anomaly_idx']].append(i)
            episodes = []
            for ep_id, idxs in ep_data.items():
                z_avg = z_all[idxs].mean(0)
                root_set = w['R_anom'][idxs[0]]
                episodes.append((z_avg, root_set))

            # LOEO: leave-one-episode-out within this seed
            for test_ep in range(len(episodes)):
                train_eps = [e for i, e in enumerate(episodes) if i != test_ep]
                gnn = SimpleGNNRanker(seed=seed)
                for A_name, A in (('true', A_true), ('pcmci', A_pcmci)):
                    gnn.fit_episode(train_eps, A)
                    # 测试: 用episode的每个窗口
                    test_idxs = ep_data[test_ep]
                    for wi in test_idxs:
                        phi = gnn.rank(z_all[wi], A)
                        hit = int(bool(set(np.argsort(-phi)[:3]) & w['R_anom'][wi]))
                        records.append(dict(
                            dataset=f'scm{seed}', seed=seed, magnitude=mag,
                            graph_source=A_name, test_episode=test_ep,
                            window=wi, hit3=hit, protocol='LOEO'))
            print(f'[E17] seed{seed} done ({len(records)} records)', flush=True)

    json.dump(records, open(os.path.join(CACHE, 'e17_gnn_positive.json'), 'w'),
              default=float)
    agg = collections.defaultdict(list)
    for r in records:
        agg[r['graph_source']].append(r['hit3'])
    print('=== GNN positive result (LOEO within-seed) ===')
    for k in sorted(agg):
        print(f'  {k}: hit@3={np.mean(agg[k]):.3f} n={len(agg[k])}')

    # 对照: 无参探针在同条件的LOEO
    from scripts.e8_round6 import gcn_rank_attribute
    probe_records = []
    for seed in range(5):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                        seed=seed, noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        A_true = s['adjacency'].astype(float)
        A_pcmci = pcmci_graph(s['normal'])['adj']
        scorer = make_scorer('iforest').fit(w['X_pool'])
        ctx = Context(w['X_pool'], scorer=scorer)
        for A_name, A in (('true', A_true), ('pcmci', A_pcmci)):
            phi = gcn_rank_attribute(w['X_anom'], ctx.with_graph(A))
            hits = M.hit_at_k(phi, w['R_anom'], 3)
            for i, h in enumerate(hits):
                probe_records.append(dict(graph_source=A_name, hit3=float(h)))
    pagg = collections.defaultdict(list)
    for r in probe_records:
        pagg[r['graph_source']].append(r['hit3'])
    print('=== Probe comparison (same conditions) ===')
    for k in sorted(pagg):
        print(f'  {k}: hit@3={np.mean(pagg[k]):.3f} n={len(pagg[k])}')


if __name__ == '__main__':
    run()
