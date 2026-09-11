# -*- coding: utf-8 -*-
"""E24: GRAA v4 on TE — 第三测试床上的方法覆盖 + 源交换轴.

图源 (与 SCM 的 source swap 对齐):
  pcmci_clean : PCMCI 在正常训练段学图 (部署场景)
  pcmci_anom  : PCMCI 在含故障段学图 (污染学习)
边置信度 |val| 来自同一次 PCMCI 运行, 与图源配对.
输出: _cache/e24_graa_te.json
"""
import os
import sys
import json
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.tep import load_tep_units
from graphs.sources import pcmci_graph
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graph_free import GRAPH_FREE
from attribution.graa_v4 import graa_v4_attribute
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')


def te_graph_with_val(series, cache_name):
    """PCMCI 学图 (含 val_matrix), 缓存 A + val."""
    path = os.path.join(CACHE, cache_name)
    if os.path.exists(path):
        d = np.load(path)
        return d['A'], d['val']
    r = pcmci_graph(series, tau_max=2)
    val = np.abs(r['val_matrix']).max(2)
    val = val / (val.max() + 1e-9)
    np.savez(path, A=r['adj'], val=val)
    return r['adj'], val


def run():
    records = []
    for u in load_tep_units():
        name = u['name']
        A_clean, val_clean = te_graph_with_val(
            u['series_normal'], f'e24_graph_clean_{name}.npz')
        A_anom, val_anom = te_graph_with_val(
            u['series_full'], f'e24_graph_anom_{name}.npz')
        print(f'{name}: clean_edges={int(A_clean.sum())} '
              f'anom_edges={int(A_anom.sum())}', flush=True)

        scorer = make_scorer('iforest').fit(u['X_pool'])
        ctx0 = Context(u['X_pool'], scorer=scorer)

        for gname, A, val in (('pcmci_clean', A_clean, val_clean),
                              ('pcmci_anom', A_anom, val_anom)):
            ctx_g = ctx0.with_graph(A)
            for pf in (0.2, 0.4):
                phi, info = graa_v4_attribute(
                    u['X_anom'], ctx_g, edge_conf=val, prune_frac=pf)
                records.append(dict(
                    dataset=name, method=f'GRAA(p={pf})', graph_source=gname,
                    hit1=float(M.mean_hit(phi, u['R_anom'], 1)),
                    hit3=float(M.mean_hit(phi, u['R_anom'], 3)),
                    n_pruned=info['n_pruned'], w_prop=info['w_prop']))
                print(f'  GRAA(p={pf}) {gname}: hit@3={records[-1]["hit3"]:.3f} '
                      f'pruned={info["n_pruned"]} w={info["w_prop"]:.2f}', flush=True)
            for mname in ('PropRank', 'DPTA-G'):
                phi_b = GRAPH_DEPENDENT[mname](u['X_anom'], ctx_g)
                records.append(dict(
                    dataset=name, method=mname, graph_source=gname,
                    hit1=float(M.mean_hit(phi_b, u['R_anom'], 1)),
                    hit3=float(M.mean_hit(phi_b, u['R_anom'], 3))))
                print(f'  {mname} {gname}: hit@3={records[-1]["hit3"]:.3f}',
                      flush=True)

        for mname in ('zDev', 'AERec'):
            phi = GRAPH_FREE[mname](u['X_anom'], ctx0)
            records.append(dict(
                dataset=name, method=mname, graph_source='none',
                hit1=float(M.mean_hit(phi, u['R_anom'], 1)),
                hit3=float(M.mean_hit(phi, u['R_anom'], 3))))
            print(f'  {mname}: hit@3={records[-1]["hit3"]:.3f}', flush=True)

    json.dump(records, open(os.path.join(CACHE, 'e24_graa_te.json'), 'w'),
              default=float)
    agg = collections.defaultdict(list)
    for r in records:
        agg[(r['method'], r['graph_source'])].append(r['hit3'])
    print('\n=== GRAA v4 on TE (6 units, hit@3 mean over units) ===')
    for k in sorted(agg, key=str):
        v = agg[k]
        print(f'  {str(k):34s} hit@3={np.mean(v):.3f} '
              f'(sd={np.std(v):.3f}, n={len(v)})')


if __name__ == '__main__':
    run()
