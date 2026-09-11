# -*- coding: utf-8 -*-
"""E35: 主表窗口级 hit 向量存储 — 支撑两级(窗口→cell)bootstrap CI.

SCM: 5 seeds x 3 scorers x 3 graph sources (graph-dependent) + graph-free.
TE:  6 units  x 3 scorers x 1 source (PCMCI) + graph-free.
只算 base 归因 (无扰动), 每窗口 hit@1/3/5 存向量.
输出: _cache/e35_wlhit.json  {key: {hit1: [...], hit3: [...], hit5: [...]}}
key = tag|dataset|scorer|method|source
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.scm import make_sample, build_windows
from data.tep import load_tep_units
from graphs.sources import pcmci_graph
from scorers import make_scorer, TorchPCAScorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graph_free import GRAPH_FREE
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')
SCORERS = ('iforest', 'pca', 'ocsvm')


def cell_rows(tag, ds, u, graphs):
    out = []
    for sk in SCORERS:
        scorer = make_scorer(sk).fit(u['X_pool'])
        torch_scorer = TorchPCAScorer(scorer) if sk == 'pca' else None
        ctx0 = Context(u['X_pool'], scorer=scorer)
        ctx0.torch_scorer = torch_scorer
        for mname, fn in GRAPH_DEPENDENT.items():
            for src_name, A in graphs.items():
                phi = fn(u['X_anom'], ctx0.with_graph(A))
                out.append((tag, ds, sk, mname, src_name, phi))
        for mname, fn in GRAPH_FREE.items():
            if mname == 'Grad' and torch_scorer is None:
                continue
            phi = fn(u['X_anom'], ctx0)
            out.append((tag, ds, sk, mname, 'none', phi))
    return out


def run():
    store = {}
    n_rec = 0

    def add(tag, ds, sk, m, src, phi, R):
        nonlocal n_rec
        k = f'{tag}|{ds}|{sk}|{m}|{src}'
        store[k] = dict(
            hit1=M.hit_at_k(phi, R, 1).astype(int).tolist(),
            hit3=M.hit_at_k(phi, R, 3).astype(int).tolist(),
            hit5=M.hit_at_k(phi, R, 5).astype(int).tolist())
        n_rec += 1

    # ---- SCM ----
    for seed in range(5):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                        seed=seed, noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        graphs = {'true': s['adjacency'].astype(float),
                  'pcmci_clean': pcmci_graph(s['normal'])['adj'],
                  'pcmci_anom': pcmci_graph(s['series'])['adj']}
        for tag, ds, sk, m, src, phi in cell_rows('scm', f'scm{seed}', w,
                                                  graphs):
            add(tag, ds, sk, m, src, phi, w['R_anom'])
        print(f'scm{seed} done ({n_rec} cells)', flush=True)

    # ---- TE ----
    for u in load_tep_units():
        cache = os.path.join(CACHE, f"te_graph_{u['name']}.npz")
        A = np.load(cache, allow_pickle=True)['A']
        for tag, ds, sk, m, src, phi in cell_rows('te', u['name'], u,
                                                  {'pcmci': A}):
            add(tag, ds, sk, m, src, phi, u['R_anom'])
        print(f"{u['name']} done ({n_rec} cells)", flush=True)

    json.dump(store, open(os.path.join(CACHE, 'e35_wlhit.json'), 'w'))
    print(f'saved e35_wlhit.json: {len(store)} cells')


if __name__ == '__main__':
    run()
