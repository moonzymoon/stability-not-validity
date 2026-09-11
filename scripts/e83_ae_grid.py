# -*- coding: utf-8 -*-
"""E83: 深度 AE scorer 下的主配置全网格 (F2 硬伤修复).

与 e1 的 medium 档同协议, 唯一区别是 scorer = AEScorer(池上训练小 AE,
可微). 目的: 回答"主网格只有经典打分器"---深度打分器下
稳定性-效度解耦图景是否复现. 输出 _cache/e83_ae_grid.json
(记录结构与 e1 兼容: method/family/strength/source/hit3/acr3_base).
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.scm import make_sample, build_windows
from graphs.sources import graph_sources
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graph_free import GRAPH_FREE
from evaluation import metrics as M
from scripts.e1_anchor import perturb_graph, perturb_input, stable_seed
from scripts.e8_round6 import AEScorer

CACHE = os.path.join(SRC, '_cache')
N_SEEDS = 5
GRAPH_STRENGTHS = {'del': [0.1, 0.2, 0.3, 0.5], 'add': [0.2, 0.5],
                   'rew': [0.2, 0.5]}
INPUT_STRENGTHS = dict(noise=[0.2, 0.5], slice=[1, 3])
M_PERTURB = 8


def run():
    records = []
    for seed in range(N_SEEDS):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                        seed=seed, noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        X_anom, X_pool, roots = w['X_anom'], w['X_pool'], w['R_anom']
        scorer = AEScorer(seed=seed).fit(X_pool)
        ctx0 = Context(X_pool, scorer=scorer)
        ctx0.torch_scorer = scorer          # Grad 需要 score_t
        sources = graph_sources(s)

        for mname, fn in GRAPH_DEPENDENT.items():
            for src_name, A_base in sources.items():
                ctx_base = ctx0.with_graph(A_base)
                phi_base = fn(X_anom, ctx_base)
                records.append(dict(
                    dataset=f'ae_scm{seed}', scorer='ae', method=mname,
                    family='none', strength=0.0, graph_source=src_name,
                    hit3=float(M.mean_hit(phi_base, roots, 3)),
                    acr3_base=None))
                for fam, strengths in GRAPH_STRENGTHS.items():
                    for st in strengths:
                        acrs = []
                        for mi in range(M_PERTURB):
                            rng = np.random.default_rng(stable_seed(
                                seed, 'ae', mname, src_name, fam, st, mi))
                            A_m = perturb_graph(A_base, fam, st, rng)
                            phi_m = fn(X_anom, ctx0.with_graph(A_m))
                            acrs.append(
                                M.acr_at_k(phi_m, phi_base, 3).mean())
                        records.append(dict(
                            dataset=f'ae_scm{seed}', scorer='ae',
                            method=mname, family=fam, strength=float(st),
                            graph_source=src_name,
                            hit3=float(M.mean_hit(phi_base, roots, 3)),
                            acr3_base=float(np.mean(acrs))))
        for mname, fn in GRAPH_FREE.items():
            phi_base = fn(X_anom, ctx0)
            records.append(dict(
                dataset=f'ae_scm{seed}', scorer='ae', method=mname,
                family='none', strength=0.0, graph_source='none',
                hit3=float(M.mean_hit(phi_base, roots, 3)),
                acr3_base=None))
            for fam, strengths in INPUT_STRENGTHS.items():
                for st in strengths:
                    acrs = []
                    for mi in range(M_PERTURB):
                        rng = np.random.default_rng(stable_seed(
                            seed, 'ae', mname, fam, st, mi))
                        X_m = perturb_input(X_anom, fam, st, rng, ctx0)
                        phi_m = fn(X_m, ctx0)
                        acrs.append(M.acr_at_k(phi_m, phi_base, 3).mean())
                    records.append(dict(
                        dataset=f'ae_scm{seed}', scorer='ae',
                        method=mname, family=fam, strength=float(st),
                        graph_source='none',
                        hit3=float(M.mean_hit(phi_base, roots, 3)),
                        acr3_base=float(np.mean(acrs))))
        print(f'[seed {seed}] records={len(records)}', flush=True)
    out = os.path.join(CACHE, 'e83_ae_grid.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=1, default=float)

    # 快速汇总: 每方法 ACR 均值 (扰动 cells) 与 base hit
    import collections
    agg = collections.defaultdict(list)
    hit = collections.defaultdict(list)
    for r in records:
        if r['family'] == 'none':
            hit[r['method']].append(r['hit3'])
        elif r['acr3_base'] is not None:
            agg[r['method']].append(r['acr3_base'])
    print('\n=== AE-scorer grid summary ===')
    for m in sorted(set(list(agg) + list(hit))):
        a = agg.get(m, [])
        print(f'  {m:14s} hit@3={np.mean(hit[m]):.2f}  '
              f'ACR@3={np.mean(a) if a else float("nan"):.2f} (n={len(a)})')
    print('->', out)


if __name__ == '__main__':
    run()
