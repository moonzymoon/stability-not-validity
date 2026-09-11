# -*- coding: utf-8 -*-
"""E25: GRAA 窗口级显著性检验 (SCM, 污染图主对比).

对每 seed 存窗口级 hit 向量 →
  (a) GRAA(p=0.4) vs PropRank / DPTA-G / zDev 的配对 bootstrap CI (Δhit@3);
  (b) 置换检验 (窗口级配对符号置换, 10k 次);
  (c) seed 级 mean±sd 表.
输出: _cache/e25_graa_signif.json
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
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graph_free import GRAPH_FREE
from attribution.graa_v4 import graa_v4_attribute
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')
BOOT = 10000


def paired_bootstrap(h_a, h_b, rng):
    """h_a, h_b: (n_w,) 同窗口配对 hit. 返回 Δ 的 95% CI."""
    d = h_a - h_b
    n = len(d)
    means = []
    for _ in range(BOOT):
        idx = rng.integers(0, n, n)
        means.append(d[idx].mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi), float(d.mean())


def paired_permutation(h_a, h_b, rng, n_perm=10000):
    """配对符号置换: 每窗口以 0.5 概率交换 a/b."""
    d = h_a - h_b
    obs = d.mean()
    n = len(d)
    signs = rng.choice([-1, 1], size=(n_perm, n))
    null = (signs * d[None, :]).mean(1)
    p = float((np.abs(null) >= abs(obs)).mean())
    return obs, p


def run():
    rng = np.random.default_rng(0)
    per_seed = {}
    for seed in range(5):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                        seed=seed, noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        A_true = s['adjacency'].astype(float)
        A_anom = pcmci_graph(s['series'])['adj']
        val_c = np.abs(pcmci_graph(s['normal'])['val_matrix']).max(2)
        val_c = val_c / (val_c.max() + 1e-9)

        scorer = make_scorer('iforest').fit(w['X_pool'])
        ctx0 = Context(w['X_pool'], scorer=scorer)
        per_seed[seed] = {}
        for gname, A in (('pcmci_anom', A_anom), ('true', A_true)):
            ctx_g = ctx0.with_graph(A)
            phi_g, info = graa_v4_attribute(w['X_anom'], ctx_g,
                                            edge_conf=val_c,
                                            prune_frac=0.4, gamma=1.0)
            hits = {'GRAA(p=0.4)': M.hit_at_k(phi_g, w['R_anom'], 3).astype(float)}
            for mname in ('PropRank', 'DPTA-G'):
                phi_b = GRAPH_DEPENDENT[mname](w['X_anom'], ctx_g)
                hits[mname] = M.hit_at_k(phi_b, w['R_anom'], 3).astype(float)
            phi_z = GRAPH_FREE['zDev'](w['X_anom'], ctx0)
            hits['zDev'] = M.hit_at_k(phi_z, w['R_anom'], 3).astype(float)
            per_seed[seed][gname] = hits
            print(f'scm{seed} {gname}: ' + ' '.join(
                f'{k}={v.mean():.3f}' for k, v in hits.items()), flush=True)

    out = {'per_seed_means': {}, 'tests': {}}
    for gname in ('pcmci_anom', 'true'):
        for k in ('GRAA(p=0.4)', 'PropRank', 'DPTA-G', 'zDev'):
            out['per_seed_means'][f'{gname}/{k}'] = [
                float(per_seed[s][gname][k].mean()) for s in range(5)]
        # 池化窗口级配对检验
        g = np.concatenate([per_seed[s][gname]['GRAA(p=0.4)'] for s in range(5)])
        for base in ('PropRank', 'DPTA-G', 'zDev'):
            b = np.concatenate([per_seed[s][gname][base] for s in range(5)])
            lo, hi, dm = paired_bootstrap(g, b, rng)
            obs, p = paired_permutation(g, b, rng)
            out['tests'][f'{gname}: GRAA vs {base}'] = dict(
                delta=dm, ci_lo=lo, ci_hi=hi, perm_p=p,
                n_windows=int(len(g)))
            print(f'{gname} GRAA vs {base}: Δ={dm:+.3f} '
                  f'[{lo:+.3f},{hi:+.3f}] p={p:.4f}', flush=True)

    json.dump(out, open(os.path.join(CACHE, 'e25_graa_signif.json'), 'w'),
              indent=1)
    print('saved e25_graa_signif.json')


if __name__ == '__main__':
    run()
