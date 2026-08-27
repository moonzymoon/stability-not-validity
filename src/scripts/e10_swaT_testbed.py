# -*- coding: utf-8 -*-
"""E10 (方案5): SWaT 第三测试床全网格 — 真实数据 + 注入式构造真值.

图源: silver(19边银标准参照图) / pcmci(正常段学图) / corr(相关性图).
方法: 3图依赖 + GCN探针 + 5图无关. 打分器: iforest / pca.
输出: _cache/e10_swat.json
"""
import os
import sys
import json
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.swat import build_swat_units
from graphs.sources import pcmci_graph
from graphs import graph_ops as go
from scorers import make_scorer, detection_auroc
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graph_free import GRAPH_FREE
from scripts.e8_round6 import gcn_rank_attribute, corr_graph
from evaluation import metrics as M
from scripts.e1_anchor import perturb_graph, stable_seed, M_PERTURB

CACHE = os.path.join(SRC, '_cache')


def main():
    records = []
    u = build_swat_units()
    print('SWaT unit: X_anom', u['X_anom'].shape, 'pool', u['X_pool'].shape,
          flush=True)
    A_silver = u['silver_adj']
    A_pcmci = pcmci_graph(u['series_normal'], tau_max=2)['adj']
    A_corr = corr_graph(u['series_normal'][:50000], thresh=0.3)
    graphs = {'silver': A_silver, 'pcmci': A_pcmci, 'corr': A_corr}
    for nm, A in graphs.items():
        r, p = go.edge_recall_precision(A, A_silver)
        print(f'  graph[{nm}] edges={int(A.sum())} vs-silver rec={r:.2f} prec={p:.2f}',
              flush=True)

    for sk in ('iforest', 'pca'):
        scorer = make_scorer(sk).fit(u['X_pool'])
        auroc = detection_auroc(scorer, u['X_pool'][::10], u['X_anom'])
        ctx0 = Context(u['X_pool'], scorer=scorer)
        from scorers import TorchPCAScorer
        if sk == 'pca':
            ctx0.torch_scorer = TorchPCAScorer(scorer)
        for mname, fn in list(GRAPH_DEPENDENT.items()) + [('GCN-Rank', gcn_rank_attribute)]:
            for gname, A in graphs.items():
                phi = fn(u['X_anom'], ctx0.with_graph(A))
                hits = dict(hit1=float(M.mean_hit(phi, u['R_anom'], 1)),
                            hit3=float(M.mean_hit(phi, u['R_anom'], 3)),
                            hit5=float(M.mean_hit(phi, u['R_anom'], 5)))
                acrs = []
                for mi in range(M_PERTURB):
                    rng = np.random.default_rng(stable_seed(
                        'e10', sk, mname, gname, mi))
                    A_m = perturb_graph(A, ['del', 'add', 'rew'][mi % 3],
                                        0.2, rng)
                    phi_m = fn(u['X_anom'], ctx0.with_graph(A_m))
                    acrs.append(M.acr_at_k(phi_m, phi, 3).mean())
                records.append(dict(
                    testbed='swat', scorer=sk, method=mname, graph_source=gname,
                    det_auroc=auroc, **hits, acr3=float(np.mean(acrs))))
                print(f"  [{sk}] {mname:13s} {gname:7s} hit@3={hits['hit3']:.2f} "
                      f"ACR={np.mean(acrs):.2f}", flush=True)
        for mname, fn in GRAPH_FREE.items():
            if mname == 'Grad' and sk != 'pca':
                continue
            phi = fn(u['X_anom'], ctx0)
            records.append(dict(
                testbed='swat', scorer=sk, method=mname, graph_source='none',
                det_auroc=auroc,
                hit1=float(M.mean_hit(phi, u['R_anom'], 1)),
                hit3=float(M.mean_hit(phi, u['R_anom'], 3)),
                hit5=float(M.mean_hit(phi, u['R_anom'], 5)),
                acr3=None))
            print(f"  [{sk}] {mname:13s} input   hit@3="
                  f"{records[-1]['hit3']:.2f}", flush=True)

    json.dump(records, open(os.path.join(CACHE, 'e10_swat.json'), 'w'),
              default=float)
    agg = collections.defaultdict(lambda: dict(h=[], a=[]))
    for r in records:
        agg[(r['method'], r['graph_source'])]['h'].append(r['hit3'])
        if r.get('acr3') is not None:
            agg[(r['method'], r['graph_source'])]['a'].append(r['acr3'])
    print('=== SWaT summary (method, source): hit3 / acr3 ===')
    for k in sorted(agg, key=str):
        h = np.mean(agg[k]['h'])
        a = np.mean(agg[k]['a']) if agg[k]['a'] else float('nan')
        print(f'  {str(k):40s} hit={h:.3f} acr={a:.3f}')


if __name__ == '__main__':
    main()
