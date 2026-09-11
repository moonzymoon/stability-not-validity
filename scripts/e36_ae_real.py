# -*- coding: utf-8 -*-
"""E36: AE 深度打分器 (GPU) 上 SWaT + RCAEval — 关闭 scorer 轴真实数据缺口.

方法: PropRank / DPTA-G / GRAA(p=0.4) (图依赖, 缓存图) + zDev.
输出: _cache/e36_ae_real.json
"""
import os
import sys
import json
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.swat import build_swat_units, silver_adj, COLIDX
from data.rcaeval import load_ob_units
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graph_free import GRAPH_FREE
from attribution.graa_v4 import graa_v4_attribute
from evaluation import metrics as M
from scripts.e8_round6 import AEScorer

CACHE = os.path.join(SRC, '_cache')


def load_swat_val():
    p = os.path.join(CACHE, 'swat_pcmci_graph.npz')
    d = np.load(p)
    val = d['val'] if 'val' in d.files else None
    if val is None:
        val = np.ones_like(d['A'])
    val = val / (val.max() + 1e-9)
    return d['A'], val


def rcaeval_graph(u, kind):
    p = os.path.join(CACHE, f"e31_graph_{kind}_{u['name']}.npz")
    d = np.load(p)
    return d['A'], d['val']


def run():
    records = []
    ae = AEScorer(seed=0)

    # ---- SWaT ----
    u = build_swat_units()
    A_pcmci, val = load_swat_val()
    silver = silver_adj()
    scorer = ae.fit(u['X_pool'])
    ctx0 = Context(u['X_pool'], scorer=scorer)
    for gname, A in (('silver', silver), ('pcmci', A_pcmci)):
        v = val if gname == 'pcmci' else np.ones_like(A)
        ctx_g = ctx0.with_graph(A)
        phi, _ = graa_v4_attribute(u['X_anom'], ctx_g, edge_conf=v,
                                   prune_frac=0.4)
        records.append(dict(testbed='swat', method='GRAA(p=0.4)',
                            graph_source=gname,
                            hit3=float(M.mean_hit(phi, u['R_anom'], 3))))
        for m in ('PropRank', 'DPTA-G'):
            phi_b = GRAPH_DEPENDENT[m](u['X_anom'], ctx_g)
            records.append(dict(testbed='swat', method=m,
                                graph_source=gname,
                                hit3=float(M.mean_hit(phi_b, u['R_anom'], 3))))
    phi_z = GRAPH_FREE['zDev'](u['X_anom'], ctx0)
    records.append(dict(testbed='swat', method='zDev', graph_source='none',
                        hit3=float(M.mean_hit(phi_z, u['R_anom'], 3))))
    print('SWaT x AE done', flush=True)

    # ---- RCAEval (cpu/mem, reps 1-3) ----
    units = [x for x in load_ob_units(reps=(1, 2, 3))
             if x['fault'] in ('cpu', 'mem')]
    for u2 in units:
        scorer = ae.fit(u2['X_pool'])
        ctx0 = Context(u2['X_pool'], scorer=scorer)
        for gname, kind in (('pcmci_clean', 'clean'), ('pcmci_anom', 'anom')):
            A, val = rcaeval_graph(u2, kind)
            ctx_g = ctx0.with_graph(A)
            phi, _ = graa_v4_attribute(u2['X_anom'], ctx_g, edge_conf=val,
                                       prune_frac=0.4)
            records.append(dict(testbed='rcaeval', unit=u2['name'],
                                method='GRAA(p=0.4)', graph_source=gname,
                                hit3=float(M.mean_hit(phi, u2['R_anom'], 3))))
            for m in ('PropRank', 'DPTA-G'):
                phi_b = GRAPH_DEPENDENT[m](u2['X_anom'], ctx_g)
                records.append(dict(testbed='rcaeval', unit=u2['name'],
                                    method=m, graph_source=gname,
                                    hit3=float(M.mean_hit(phi_b,
                                                         u2['R_anom'], 3))))
        phi_z = GRAPH_FREE['zDev'](u2['X_anom'], ctx0)
        records.append(dict(testbed='rcaeval', unit=u2['name'], method='zDev',
                            graph_source='none',
                            hit3=float(M.mean_hit(phi_z, u2['R_anom'], 3))))
        print(f"{u2['name']} done", flush=True)

    json.dump(records, open(os.path.join(CACHE, 'e36_ae_real.json'), 'w'),
              default=float)
    agg = collections.defaultdict(list)
    for r in records:
        agg[(r['testbed'], r['method'], r['graph_source'])].append(r['hit3'])
    print('\n=== AE scorer on real testbeds (hit@3) ===')
    for k in sorted(agg, key=str):
        print(f'  {str(k):44s} {np.mean(agg[k]):.3f}')


if __name__ == '__main__':
    run()
