# -*- coding: utf-8 -*-
"""E29: GRAA 在关系型异常下的行为 (方法评估缺口的补齐).

动机: 关系型异常下 zDev 崩溃 (排名反转发现), 而 GRAA 集成含 zDev 分量.
问题: GRAA 在关系型场景是优雅退化还是继承崩溃?
设计: 与 e6 关系型窗口同源 (make_relation_sample), 5 seeds,
图源 true/pcmci_clean/pcmci_anom, GRAA(p=0.2,0.4) vs PropRank/DPTA-G/zDev/AERec.
输出: _cache/e29_graa_rel.json
"""
import os
import sys
import json
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from scripts.e5_review import make_relation_sample
from data.scm import build_windows
from graphs.sources import pcmci_graph
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graph_free import GRAPH_FREE
from attribution.graa_v4 import graa_v4_attribute
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')


def run():
    records = []
    for seed in range(5):
        s = make_relation_sample(seed=seed)
        w = build_windows(s)
        A_true = s['adjacency'].astype(float)
        A_clean = pcmci_graph(s['normal'])['adj']
        A_anom = pcmci_graph(s['series'])['adj']
        val_c = np.abs(pcmci_graph(s['normal'])['val_matrix']).max(2)
        val_c = val_c / (val_c.max() + 1e-9)

        scorer = make_scorer('iforest').fit(w['X_pool'])
        ctx0 = Context(w['X_pool'], scorer=scorer)

        for gname, A in (('true', A_true), ('pcmci_clean', A_clean),
                         ('pcmci_anom', A_anom)):
            ctx_g = ctx0.with_graph(A)
            for pf in (0.2, 0.4):
                phi, info = graa_v4_attribute(
                    w['X_anom'], ctx_g, edge_conf=val_c, prune_frac=pf)
                records.append(dict(
                    dataset=f'rel{seed}', method=f'GRAA(p={pf})',
                    graph_source=gname,
                    hit3=float(M.mean_hit(phi, w['R_anom'], 3))))
                print(f'  rel{seed} GRAA(p={pf}) {gname}: '
                      f'hit@3={records[-1]["hit3"]:.3f} w={info["w_prop"]:.2f}',
                      flush=True)
            for mname in ('PropRank', 'DPTA-G'):
                phi_b = GRAPH_DEPENDENT[mname](w['X_anom'], ctx_g)
                records.append(dict(
                    dataset=f'rel{seed}', method=mname, graph_source=gname,
                    hit3=float(M.mean_hit(phi_b, w['R_anom'], 3))))
        for mname in ('zDev', 'AERec'):
            phi = GRAPH_FREE[mname](w['X_anom'], ctx0)
            records.append(dict(
                dataset=f'rel{seed}', method=mname, graph_source='none',
                hit3=float(M.mean_hit(phi, w['R_anom'], 3))))
        print(f'rel{seed} done: zDev={records[-2]["hit3"]:.3f} '
              f'AERec={records[-1]["hit3"]:.3f}', flush=True)

    json.dump(records, open(os.path.join(CACHE, 'e29_graa_rel.json'), 'w'),
              default=float)
    agg = collections.defaultdict(list)
    for r in records:
        agg[(r['method'], r['graph_source'])].append(r['hit3'])
    print('\n=== GRAA under RELATIONAL anomalies (hit@3, 5 seeds) ===')
    for k in sorted(agg, key=str):
        print(f'  {str(k):36s} {np.mean(agg[k]):.3f}')


if __name__ == '__main__':
    run()
