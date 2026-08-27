# -*- coding: utf-8 -*-
"""E40: TE 协议稳健性 — 窗口步距加密 + 窗口长度变化.

变体:
  base : w=16, stride=60  (与 e24 相同, 复现参照)
  dense: w=16, stride=12  (窗口数约 x5, 检验单元内稳定性)
  w8   : w=8,  stride=60
  w32  : w=32, stride=60
方法: PropRank / DPTA-G (两种图源), GRAA(p=0.4) (两种图源), zDev / AERec.
图学习缓存复用 e24 的 PCMCI 图 (与窗口化无关).
输出: _cache/e40_te_dense.json  (逐单元 hit@3 + 分层 bootstrap CI)
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
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graph_free import GRAPH_FREE
from attribution.graa_v4 import graa_v4_attribute
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')

VARIANTS = {'base': (16, 60), 'dense': (16, 12), 'w8': (8, 60),
            'w32': (32, 60)}


def te_graph(name, kind):
    path = os.path.join(CACHE, f'e24_graph_{kind}_{name}.npz')
    d = np.load(path)
    return d['A'], d['val']


def run():
    records = []
    for vname, (w, st) in VARIANTS.items():
        units = load_tep_units(window=w, stride=st)
        for u in units:
            name = u['name']
            scorer = make_scorer('iforest').fit(u['X_pool'])
            ctx0 = Context(u['X_pool'], scorer=scorer)
            A_clean, val_c = te_graph(name, 'clean')
            A_anom, val_a = te_graph(name, 'anom')
            rows = []
            for gname, A, val in (('pcmci_clean', A_clean, val_c),
                                  ('pcmci_anom', A_anom, val_a)):
                ctx_g = ctx0.with_graph(A)
                phi, _ = graa_v4_attribute(u['X_anom'], ctx_g, edge_conf=val,
                                           prune_frac=0.4)
                rows.append(dict(method='GRAA(p=0.4)', graph_source=gname,
                                 hit3=float(M.mean_hit(phi, u['R_anom'], 3))))
                for mname in ('PropRank', 'DPTA-G'):
                    phi_b = GRAPH_DEPENDENT[mname](u['X_anom'], ctx_g)
                    rows.append(dict(method=mname, graph_source=gname,
                                     hit3=float(M.mean_hit(phi_b, u['R_anom'],
                                                           3))))
            for mname in ('zDev', 'AERec'):
                phi = GRAPH_FREE[mname](u['X_anom'], ctx0)
                rows.append(dict(method=mname, graph_source='none',
                                 hit3=float(M.mean_hit(phi, u['R_anom'], 3))))
            for r in rows:
                r.update(variant=vname, unit=name,
                         n_windows=int(len(u['X_anom'])))
                records.append(r)
            print(f'[{vname}] {name} n={len(u["X_anom"])} '
                  f'GRAA_clean_hit3={rows[0]["hit3"]:.3f}', flush=True)
    json.dump(records, open(os.path.join(CACHE, 'e40_te_dense.json'), 'w'),
              default=float)

    print('\n=== hit@3 by variant x method (mean over 6 units) ===')
    agg = collections.defaultdict(list)
    for r in records:
        agg[(r['variant'], r['method'], r['graph_source'])].append(r['hit3'])
    for k in sorted(agg, key=str):
        v = agg[k]
        print(f'  {str(k):46s} {np.mean(v):.3f} (sd {np.std(v):.3f})')


if __name__ == '__main__':
    run()
