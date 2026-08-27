# -*- coding: utf-8 -*-
"""E46: GRAA vs PropRank 的配对效应量 — Cliff's delta (窗口级, 污染图).

同时复算 mean hit3 作为回归校验 (scm0 true GRAA 应=0.686).
输出: _cache/e46_effect.json
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.scm import make_sample, build_windows
from graphs.sources import graph_sources, pcmci_graph
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graa_v4 import graa_v4_attribute
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')


def val_conf_e33(series_normal):
    """与 e33 完全一致的置信度: 正常段 PCMCI |val| 全矩阵 / max (不加 adj 罩)."""
    r = pcmci_graph(series_normal)
    v = np.abs(r['val_matrix']).max(2)
    return v / (v.max() + 1e-9)


def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    return (gt - lt) / (len(x) * len(y))


def run():
    out = {}
    for gname in ('pcmci_anom', 'true'):
        g_hits, p_hits, g_means, p_means = [], [], [], []
        for seed in range(5):
            s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                            seed=seed, noise_std=0.5, magnitude=0.5)
            w = build_windows(s)
            src = graph_sources(s)
            A = src[gname].astype(float)
            val = val_conf_e33(s['normal'])
            scorer = make_scorer('iforest').fit(w['X_pool'])
            ctx = Context(w['X_pool'], scorer=scorer).with_graph(A)
            phi_g, _ = graa_v4_attribute(w['X_anom'], ctx, edge_conf=val,
                                         prune_frac=0.4)
            phi_p = GRAPH_DEPENDENT['PropRank'](w['X_anom'], ctx)
            hg = M.hit_at_k(phi_g, w['R_anom'], 3)
            hp = M.hit_at_k(phi_p, w['R_anom'], 3)
            g_hits += list(hg)
            p_hits += list(hp)
            g_means.append(float(hg.mean()))
            p_means.append(float(hp.mean()))
            print(f'{gname} seed{seed}: GRAA={hg.mean():.3f} '
                  f'PropRank={hp.mean():.3f}', flush=True)
        out[gname] = dict(
            graa_mean=float(np.mean(g_means)),
            proprank_mean=float(np.mean(p_means)),
            cliffs_delta=round(float(cliffs_delta(g_hits, p_hits)), 3),
            n_windows=len(g_hits))
        print(gname, out[gname], flush=True)
    json.dump(out, open(os.path.join(CACHE, 'e46_effect.json'), 'w'))


if __name__ == '__main__':
    run()
