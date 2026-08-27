# -*- coding: utf-8 -*-
"""E49v2: 种子 0-9 全量重跑 — 配对窗口级置换检验 + 种子级 Wilcoxon.

e33 口径; 输出逐种子均值与窗口命中向量 -> e49_seeds10.json + e49_summary.json
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import sys
import json
import numpy as np
from scipy.stats import wilcoxon

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


def cliffs(x, y):
    x, y = np.asarray(x), np.asarray(y)
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    return (gt - lt) / (len(x) * len(y))


def perm_p(diff, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    obs = abs(diff.mean())
    cnt = 0
    for _ in range(n):
        s = rng.choice([-1, 1], size=len(diff))
        if abs((diff * s).mean()) >= obs - 1e-12:
            cnt += 1
    return cnt / n


def run():
    raw = {}
    summary = {}
    for gname in ('pcmci_anom', 'true'):
        g_m, p_m, g_h, p_h = [], [], [], []
        for seed in range(10):
            s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                            seed=seed, noise_std=0.5, magnitude=0.5)
            w = build_windows(s)
            src = graph_sources(s)
            A = src[gname].astype(float)
            r = pcmci_graph(s['normal'])
            val = np.abs(r['val_matrix']).max(2)
            val = val / (val.max() + 1e-9)
            scorer = make_scorer('iforest').fit(w['X_pool'])
            ctx = Context(w['X_pool'], scorer=scorer).with_graph(A)
            phi_g, _ = graa_v4_attribute(w['X_anom'], ctx, edge_conf=val,
                                         prune_frac=0.4)
            phi_p = GRAPH_DEPENDENT['PropRank'](w['X_anom'], ctx)
            hg = M.hit_at_k(phi_g, w['R_anom'], 3)
            hp = M.hit_at_k(phi_p, w['R_anom'], 3)
            g_m.append(float(hg.mean()))
            p_m.append(float(hp.mean()))
            g_h += list(hg)
            p_h += list(hp)
            print(f'{gname} seed{seed}: G={hg.mean():.3f} P={hp.mean():.3f}',
                  flush=True)
        G, P = np.array(g_m), np.array(p_m)
        gh, ph = np.array(g_h), np.array(p_h)
        try:
            _, wp = wilcoxon(G, P)
        except Exception:
            wp = float('nan')
        summary[gname] = dict(
            graa_seed_mean=float(G.mean()), prop_seed_mean=float(P.mean()),
            seeds_positive=int((G > P).sum()), n_seeds=10,
            wilcoxon_p=float(wp), delta_pp=float((G - P).mean() * 100),
            window_delta_pp=float((gh - ph).mean() * 100),
            window_perm_p=perm_p(gh - ph), cliffs_delta=float(cliffs(gh, ph)),
            n_windows=int(len(gh)))
        raw[gname] = dict(seed_means_g=g_m, seed_means_p=p_m,
                          hits_g=[int(x) for x in gh], hits_p=[int(x) for x in ph])
        print(gname, summary[gname], flush=True)
    json.dump(raw, open(os.path.join(CACHE, 'e49_seeds10.json'), 'w'))
    json.dump(summary, open(os.path.join(CACHE, 'e49_summary.json'), 'w'))
    print('-> e49_seeds10.json / e49_summary.json')


if __name__ == '__main__':
    run()
