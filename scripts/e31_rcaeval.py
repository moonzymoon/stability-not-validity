# -*- coding: utf-8 -*-
"""E31: RCAEval RE1-OB 第四测试床 — 真实分布式系统 + 注入式构造真值.

5 服务 x {cpu, mem} 故障 x 3 次重复 = 30 单元.
图源: pcmci_clean (注入前段学图) vs pcmci_anom (全程含故障学图) — 源交换轴.
方法: GRAA(0.2,0.4) / PropRank / DPTA-G (图) + zDev / AERec / Random (免图).
输出: _cache/e31_rcaeval.json
"""
import os
import sys
import json
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.rcaeval import load_ob_units
from graphs.sources import pcmci_graph
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graph_free import GRAPH_FREE
from attribution.graa_v4 import graa_v4_attribute
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')


def graph_with_val(series, cache_name):
    path = os.path.join(CACHE, cache_name)
    if os.path.exists(path):
        d = np.load(path)
        return d['A'], d['val']
    r = pcmci_graph(series, tau_max=1)
    val = np.abs(r['val_matrix']).max(2)
    val = val / (val.max() + 1e-9)
    np.savez(path, A=r['adj'], val=val)
    return r['adj'], val


def run():
    units = load_ob_units(reps=(1, 2, 3))
    # 只要 cpu/mem (注入在中点, 正常段充足)
    units = [u for u in units if u['fault'] in ('cpu', 'mem')]
    print(f'{len(units)} units (cpu/mem, reps 1-3)', flush=True)

    records = []
    for u in units:
        A_clean, val_clean = graph_with_val(
            u['series_normal'], f"e31_graph_clean_{u['name']}.npz")
        A_anom, val_anom = graph_with_val(
            u['series_full'], f"e31_graph_anom_{u['name']}.npz")

        scorer = make_scorer('iforest').fit(u['X_pool'])
        ctx0 = Context(u['X_pool'], scorer=scorer)

        for gname, A, val in (('pcmci_clean', A_clean, val_clean),
                              ('pcmci_anom', A_anom, val_anom)):
            ctx_g = ctx0.with_graph(A)
            for pf in (0.2, 0.4):
                phi, info = graa_v4_attribute(
                    u['X_anom'], ctx_g, edge_conf=val, prune_frac=pf)
                records.append(dict(
                    dataset=u['name'], method=f'GRAA(p={pf})',
                    graph_source=gname,
                    hit1=float(M.mean_hit(phi, u['R_anom'], 1)),
                    hit3=float(M.mean_hit(phi, u['R_anom'], 3)),
                    n_pruned=info['n_pruned']))
            for mname in ('PropRank', 'DPTA-G'):
                phi_b = GRAPH_DEPENDENT[mname](u['X_anom'], ctx_g)
                records.append(dict(
                    dataset=u['name'], method=mname, graph_source=gname,
                    hit1=float(M.mean_hit(phi_b, u['R_anom'], 1)),
                    hit3=float(M.mean_hit(phi_b, u['R_anom'], 3))))
        for mname in ('zDev', 'AERec', 'Random'):
            phi = GRAPH_FREE[mname](u['X_anom'], ctx0)
            records.append(dict(
                dataset=u['name'], method=mname, graph_source='none',
                hit1=float(M.mean_hit(phi, u['R_anom'], 1)),
                hit3=float(M.mean_hit(phi, u['R_anom'], 3))))
        print(f"{u['name']}: done "
              f"(zDev hit3={records[-3]['hit3']:.3f} "
              f"PropRank clean={next(r['hit3'] for r in records if r['dataset']==u['name'] and r['method']=='PropRank' and r['graph_source']=='pcmci_clean'):.3f})",
              flush=True)

    json.dump(records, open(os.path.join(CACHE, 'e31_rcaeval.json'), 'w'),
              default=float)
    agg = collections.defaultdict(list)
    for r in records:
        agg[(r['method'], r['graph_source'])].append(r['hit3'])
    print('\n=== RCAEval OB (30 units, hit@3 mean) ===')
    for k in sorted(agg, key=str):
        v = agg[k]
        print(f'  {str(k):34s} {np.mean(v):.3f} (sd={np.std(v):.3f})')


if __name__ == '__main__':
    run()
