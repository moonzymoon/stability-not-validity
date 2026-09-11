# -*- coding: utf-8 -*-
"""E37: 学习式边置信度 — SCM 训练集构建 (边级特征 + 真图标签).

对每 seed x magnitude x 图源(正常学习/污染学习):
  学图的每条边 (i,j): 特征 = [|val|, 出度, 入度, 图密度, stability]
  标签 = 是否在真图中 (1=真边, 0=伪边)
stability: K=4 次 50% 池子采样重跑 PCMCI, 边出现比例.
输出: _cache/e37_edges.json
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.scm import make_sample, build_windows
from graphs.sources import pcmci_graph

CACHE = os.path.join(SRC, '_cache')
MAGS = (0.25, 0.5, 1.5)
K_BOOT = 4


def edge_features(A, val, stable, dens):
    outd = A.sum(1)
    ind = A.sum(0)
    rows = []
    ii, jj = np.where(A > 0)
    for i, j in zip(ii, jj):
        rows.append(dict(i=int(i), j=int(j), feat=[
            float(val[i, j]), float(outd[i]), float(ind[j]),
            float(dens), float(stable[i, j])]))
    return rows


def run():
    rng = np.random.default_rng(0)
    records = []
    for mag in MAGS:
        for seed in range(5):
            s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                            seed=seed, noise_std=0.5, magnitude=mag)
            A_true = s['adjacency'].astype(bool)
            normal = s['normal']
            series = s['series']
            for gname, data in (('clean', normal), ('anom', series)):
                r = pcmci_graph(data)
                A, val = r['adj'], np.abs(r['val_matrix']).max(2)
                val = val / (val.max() + 1e-9)
                dens = float(A.mean())
                # bootstrap stability
                stable = np.zeros_like(A, dtype=float)
                T = len(data)
                for k in range(K_BOOT):
                    idx = np.sort(rng.choice(T, T // 2, replace=False))
                    rk = pcmci_graph(data[idx])
                    stable += (rk['adj'] > 0)
                stable /= K_BOOT
                for e in edge_features(A, val, stable, dens):
                    lab = bool(A_true[e['i'], e['j']])
                    records.append(dict(
                        mag=mag, seed=seed, graph=gname, **e, label=int(lab)))
            print(f'mag{mag} seed{seed} done ({len(records)} edges)',
                  flush=True)
    json.dump(records, open(os.path.join(CACHE, 'e37_edges.json'), 'w'))
    labs = [r['label'] for r in records]
    print(f'saved e37_edges.json: {len(records)} edges, '
          f'pos-rate={np.mean(labs):.3f}')


if __name__ == '__main__':
    run()
