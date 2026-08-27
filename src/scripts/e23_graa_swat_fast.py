# -*- coding: utf-8 -*-
"""E23: GRAA v4 on SWaT — 使用缓存的图，快速计算."""
import os, sys, json, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.swat import build_swat_units, silver_adj, COLIDX
from graphs.sources import pcmci_graph
from graphs import graph_ops as go
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graph_free import GRAPH_FREE
from attribution.graa_v4 import graa_v4_attribute
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')


def run():
    u = build_swat_units()
    print(f'SWaT: X_anom={u["X_anom"].shape}, pool={u["X_pool"].shape}')

    # 尝试加载缓存的图
    pcmci_cache = os.path.join(CACHE, 'te_graph_swar_pcmci.npz')
    silver = silver_adj()

    # PCMCI在SWaT正常段学图（缓存）
    pcmci_cache_swar = os.path.join(CACHE, 'swat_pcmci_graph.npz')
    if os.path.exists(pcmci_cache_swar):
        A_pcmci = np.load(pcmci_cache_swar)['A']
        val_c = np.load(pcmci_cache_swar)['val'] if 'val' in np.load(pcmci_cache_swar) else None
        print('Loaded cached SWaT PCMCI graph')
    else:
        # 用正常段前10k步（减少计算量）
        print('Computing PCMCI on SWaT (10k steps)...', flush=True)
        r = pcmci_graph(u['series_normal'][:10000], tau_max=1)
        A_pcmci = r['adj']
        val_c = np.abs(r['val_matrix']).max(2)
        val_c = val_c / (val_c.max() + 1e-9)
        np.savez(pcmci_cache_swar, A=A_pcmci, val=val_c)
        print(f'  edges={int(A_pcmci.sum())}')

    # 归一化置信度
    if val_c is not None:
        val_c = val_c / (val_c.max() + 1e-9)

    scorer = make_scorer('iforest').fit(u['X_pool'])
    ctx0 = Context(u['X_pool'], scorer=scorer)
    records = []

    for gname, A in [('silver', silver), ('pcmci', A_pcmci)]:
        ctx_g = ctx0.with_graph(A)

        # GRAA v4
        for pf in (0.2, 0.4):
            phi, info = graa_v4_attribute(u['X_anom'], ctx_g,
                                           edge_conf=val_c, prune_frac=pf)
            h3 = M.mean_hit(phi, u['R_anom'], 3)
            h1 = M.mean_hit(phi, u['R_anom'], 1)
            records.append(dict(method=f'GRAA-v4(p={pf})', graph_source=gname,
                               hit1=h1, hit3=h3,
                               n_pruned=info['n_pruned']))
            print(f'  GRAA(p={pf}) {gname}: hit@3={h3:.3f} pruned={info["n_pruned"]}',
                  flush=True)

        # 基线
        for mname in ('PropRank', 'DPTA-G'):
            phi_b = GRAPH_DEPENDENT[mname](u['X_anom'], ctx_g)
            records.append(dict(method=mname, graph_source=gname,
                               hit3=M.mean_hit(phi_b, u['R_anom'], 3)))
            print(f'  {mname} {gname}: {records[-1]["hit3"]:.3f}')

    for mname in ('zDev', 'AERec'):
        phi = GRAPH_FREE[mname](u['X_anom'], ctx0)
        records.append(dict(method=mname, graph_source='none',
                           hit3=M.mean_hit(phi, u['R_anom'], 3)))
        print(f'  {mname}: {records[-1]["hit3"]:.3f}')

    json.dump(records, open(os.path.join(CACHE, 'e23_graa_swat.json'), 'w'),
              default=float)
    agg = collections.defaultdict(list)
    for r in records:
        agg[(r['method'], r['graph_source'])].append(r['hit3'])
    print('\n=== GRAA v4 on SWaT (hit@3) ===')
    for k in sorted(agg, key=str):
        print(f'  {str(k):38s} hit={np.mean(agg[k]):.3f}')


if __name__ == '__main__':
    run()
