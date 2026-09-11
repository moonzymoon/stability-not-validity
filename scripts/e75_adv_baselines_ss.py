# -*- coding: utf-8 -*-
"""E75: 对抗威胁模型补全 — degree-top / val-top 同预算基线 (SS, port e70).

回应元宝⑩: 贪心搜索之外, 补两条平凡基线同预算对比,
排除"贪心 = 删最重要边"的解释:
  degree-top: 删累计度数最高的 B 条边
  val-top:    删 PCMCI 置信度最高的 B 条边
单元/方法/协议与 e70 完全一致 (6 cpu 单元, B=2, iforest)。
输出: _cache/e75_adv_baselines_ss.json
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
import sys
import json
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.rcaeval2 import load_units
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')
B = 2


def topk_sets(phi, K=3):
    return [set(np.argsort(-row)[:K]) for row in phi]


def acr_vs_base(phi_m, phi0):
    return float(M.acr_at_k(phi_m, phi0, 3).mean())


def degree_edges(A):
    d = A.shape[0]
    deg = A.sum(0) + A.sum(1)          # 无向累计度
    edges = [(i, j) for i in range(d) for j in range(d) if A[i, j] > 0]
    edges.sort(key=lambda e: -(deg[e[0]] + deg[e[1]]))
    return edges


def val_edges(A, val):
    d = A.shape[0]
    edges = [(i, j) for i in range(d) for j in range(d) if A[i, j] > 0]
    edges.sort(key=lambda e: -float(val[e[0], e[1]]))
    return edges


def apply_del(A, edges):
    Bm = A.copy()
    for i, j in edges[:B]:
        Bm[i, j] = 0
    return Bm


def run():
    units_all = [u for u in load_units('ss', reps=(1,))
                 if not np.isnan(u['series_full']).any()]
    by_svc = {}
    for u in units_all:
        if u['fault'] == 'cpu' and u['service'] not in by_svc:
            by_svc[u['service']] = u
    units = [by_svc[s] for s in sorted(by_svc)][:6]
    records = []
    for ui, u in enumerate(units):
        d = np.load(os.path.join(CACHE, f'e66_graph_clean_{u["name"]}.npz'))
        A, val = d['A'].astype(float), d['val']
        scorer = make_scorer('iforest').fit(u['X_pool'])
        ctx0 = Context(u['X_pool'], scorer=scorer)
        for mname in ('PropRank', 'DPTA-G'):
            fn = GRAPH_DEPENDENT[mname]
            phi0 = fn(u['X_anom'], ctx0.with_graph(A))
            rec = dict(unit=u['name'], method=mname,
                       degree_top=acr_vs_base(
                           fn(u['X_anom'], ctx0.with_graph(
                               apply_del(A, degree_edges(A)))), phi0),
                       val_top=acr_vs_base(
                           fn(u['X_anom'], ctx0.with_graph(
                               apply_del(A, val_edges(A, val)))), phi0))
            records.append(rec)
            print(f'[{ui+1}/{len(units)}] {u["name"]} {mname}: '
                  f'deg-top={rec["degree_top"]:.3f} '
                  f'val-top={rec["val_top"]:.3f}', flush=True)
    json.dump(records, open(os.path.join(CACHE, 'e75_adv_baselines_ss.json'),
                            'w'), default=float, indent=1)
    import collections
    agg = collections.defaultdict(list)
    for r in records:
        for k in ('degree_top', 'val_top'):
            agg[(r['method'], k)].append(r[k])
    # 对照: e70 的贪心与随机
    e70 = json.load(open(os.path.join(CACHE, 'e70_adversarial_ss.json')))
    for r in e70:
        agg[(r['method'], 'greedy_B2')].append(r['adversarial_B2'])
        agg[(r['method'], 'random_q0.1')].append(r['random_q0.1'])
    print('\n=== E75 威胁模型基线 (SS, B=2, mean) ===')
    for k in sorted(agg, key=str):
        print(f'  {str(k):30} {np.mean(agg[k]):.3f}')


if __name__ == '__main__':
    run()
