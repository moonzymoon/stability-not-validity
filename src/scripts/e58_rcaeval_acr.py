# -*- coding: utf-8 -*-
"""E58: 完备1 — RCAEval 第四测试床补算 ACR@3 (双指标在四床闭环).
对 e31 的 30 单元 (cpu/mem × 3 reps), 以 pcmci_clean / pcmci_anom 学图为 base,
施加 del/add/rew 图扰动 (q=0.1, M=8 循环), 计算 PropRank/DPTA-G/GRAA(p=0.4)
的 ACR@3; hit@3 沿用 e31 (重算以自洽). 图缓存复用 e31_graph_*.npz.
输出: _cache/e58_rcaeval_acr.json
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
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
from attribution.graa_v4 import graa_v4_attribute
from evaluation import metrics as M
from scripts.e1_anchor import perturb_graph, stable_seed

CACHE = os.path.join(SRC, '_cache')
Q = 0.1
M_INST = 8


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


def topk(phi, K=3):
    return np.argsort(-phi, axis=1)[:, :K]


def acr(phi0, phis, K=3):
    t0 = topk(phi0, K)
    ov = []
    for pm in phis:
        tm = topk(pm, K)
        ov.append(np.mean([len(set(a) & set(b)) / K
                           for a, b in zip(t0, tm)]))
    return float(np.mean(ov))


def run():
    units = [u for u in load_ob_units(reps=(1, 2, 3))
             if u['fault'] in ('cpu', 'mem')]
    print(f'{len(units)} units', flush=True)
    records = []
    for u in units:
        A_clean, val_clean = graph_with_val(
            u['series_normal'], f"e31_graph_clean_{u['name']}.npz")
        A_anom, val_anom = graph_with_val(
            u['series_full'], f"e31_graph_anom_{u['name']}.npz")
        scorer = make_scorer('iforest').fit(u['X_pool'])
        ctx0 = Context(u['X_pool'], scorer=scorer)
        for gname, A, val in (('pcmci_clean', A_clean, val_anom),
                              ('pcmci_anom', A_anom, val_anom)):
            ctx_g = ctx0.with_graph(A)
            jobs = [('PropRank', lambda: GRAPH_DEPENDENT['PropRank'](
                u['X_anom'], ctx_g)),
                    ('DPTA-G', lambda: GRAPH_DEPENDENT['DPTA-G'](
                        u['X_anom'], ctx_g)),
                    ('GRAA(p=0.4)', lambda: graa_v4_attribute(
                        u['X_anom'], ctx_g, edge_conf=val,
                        prune_frac=0.4)[0])]
            for mname, base_fn in jobs:
                phi0 = base_fn()
                phis = []
                for mi in range(M_INST):
                    rng = np.random.default_rng(stable_seed(
                        'e58', u['name'], gname, mname, mi))
                    A_m = perturb_graph(A, ['del', 'add', 'rew'][mi % 3],
                                        Q, rng)
                    ctx_m = ctx0.with_graph(A_m)
                    if mname.startswith('GRAA'):
                        phis.append(graa_v4_attribute(
                            u['X_anom'], ctx_m, edge_conf=val,
                            prune_frac=0.4)[0])
                    else:
                        phis.append(GRAPH_DEPENDENT[mname](
                            u['X_anom'], ctx_m))
                records.append(dict(
                    unit=u['name'], method=mname, graph_source=gname,
                    hit3=float(M.mean_hit(phi0, u['R_anom'], 3)),
                    acr3=acr(phi0, phis)))
            print(f"{u['name']} {gname}: "
                  + ' '.join(f"{r['method']}={r['acr3']:.2f}"
                             for r in records[-3:]), flush=True)
    json.dump(records, open(os.path.join(CACHE, 'e58_rcaeval_acr.json'),
                            'w'), default=float)
    agg = collections.defaultdict(list)
    for r in records:
        agg[(r['method'], r['graph_source'])].append((r['acr3'], r['hit3']))
    print('\n=== RCAEval ACR@3 (30 units, q=0.1, M=8) ===')
    for k in sorted(agg, key=str):
        v = agg[k]
        print(f'  {str(k):30s} acr3={np.mean([x[0] for x in v]):.3f} '
              f'hit3={np.mean([x[1] for x in v]):.3f} (n={len(v)})')


if __name__ == '__main__':
    run()
