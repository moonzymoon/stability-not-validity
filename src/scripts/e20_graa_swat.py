# -*- coding: utf-8 -*-
"""E20: GRAA在SWaT上验证."""
import os, sys, json, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.swat import build_swat_units
from graphs.sources import pcmci_graph
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graph_free import GRAPH_FREE
from attribution.graa_v2 import graa_v2_attribute
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')

def run():
    u = build_swat_units()
    A_silver = u['silver_adj']
    A_pcmci = pcmci_graph(u['series_normal'], tau_max=2)['adj']
    scorer = make_scorer('iforest').fit(u['X_pool'])
    ctx0 = Context(u['X_pool'], scorer=scorer)
    records = []

    for gname, A in [('silver', A_silver), ('pcmci', A_pcmci)]:
        ctx_g = ctx0.with_graph(A)
        for gamma in (0.5, 1.0, 2.0):
            phi, info = graa_v2_attribute(u['X_anom'], ctx_g, gamma=gamma)
            h3 = M.mean_hit(phi, u['R_anom'], 3)
            h1 = M.mean_hit(phi, u['R_anom'], 1)
            records.append(dict(method=f'GRAA(g={gamma})', graph_source=gname,
                               hit1=h1, hit3=h3,
                               w_prop=info['w_prop_mean']))
            print(f'  GRAA(g={gamma}) {gname}: hit@3={h3:.3f} w={info["w_prop_mean"]:.2f}',
                  flush=True)

        # 基线对照
        for mname in ('PropRank', 'DPTA-G'):
            phi = GRAPH_DEPENDENT[mname](u['X_anom'], ctx_g)
            records.append(dict(method=mname, graph_source=gname,
                               hit3=M.mean_hit(phi, u['R_anom'], 3)))
        print(f'  PropRank {gname}: {records[-1]["hit3"]:.3f}')

    for mname in ('zDev', 'AERec'):
        phi = GRAPH_FREE[mname](u['X_anom'], ctx0)
        records.append(dict(method=mname, graph_source='none',
                           hit3=M.mean_hit(phi, u['R_anom'], 3)))

    json.dump(records, open(os.path.join(CACHE, 'e20_graa_swat.json'), 'w'),
              default=float)
    agg = collections.defaultdict(list)
    for r in records:
        agg[(r['method'], r['graph_source'])].append(r['hit3'])
    print('\n=== GRAA on SWaT (hit@3) ===')
    for k in sorted(agg, key=str):
        print(f'  {str(k):38s} hit={np.mean(agg[k]):.3f}')

if __name__ == '__main__':
    run()
