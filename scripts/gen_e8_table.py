# -*- coding: utf-8 -*-
"""E8 结果表: 现代图方法 + 第三图源 + 深度打分器."""
import json
import os
import collections

import numpy as np

CACHE = '_cache'
TAB = os.path.join('..', 'paper', 'tables')
BSB = chr(92)

recs = json.load(open(os.path.join(CACHE, 'e8_grid.json'), encoding='utf-8'))
agg = collections.defaultdict(lambda: dict(h=[], a=[]))
for r in recs:
    k = (r['method'], r['graph_source'])
    agg[k]['h'].append(r['hit3'])
    if r.get('acr3') is not None:
        agg[k]['a'].append(r['acr3'])

L = [BSB + 'begin{table}[t]' + BSB + 'centering' + BSB + 'small',
     BSB + 'caption{Method-representation and source-robustness extensions '
     '(SCM medium, 5 trajectories; hit@3 of the base-graph attribution and '
     'ACR@3 under graph perturbation, averaged over the three scorers): '
     '(i) a parameter-free GCN-style propagation ranker (reversed-graph '
     'aggregation, two layers) behaves like the other genuinely '
     'graph-transmitting method --- its validity tracks graph quality '
     '(correlation-graph $-36$ points); (ii) the correlation-threshold '
     'graph (edge recall 0.00--0.10) is a third, practitioner-common '
     'systematic bias source and costs PropRank 31 points; (iii) under the '
     'deep AE scorer the propagation ranker is unchanged (these methods do '
     'not consume scorer output) and gradients on the AE scorer remain in '
     'the same band as on PCA.}',
     BSB + 'label{tab:e8}',
     BSB + 'begin{tabular}{llcc}', BSB + 'toprule',
     'Method & Graph source & hit@3 & ACR@3' + BSB + BSB, BSB + 'midrule']
order = [('GCN-Rank', 'true'), ('GCN-Rank', 'pcmci_anom'), ('GCN-Rank', 'corr'),
         ('PropRank', 'true'), ('PropRank', 'pcmci_anom'), ('PropRank', 'corr')]
srcname = {'true': 'true', 'pcmci_anom': 'PCMCI-anom', 'corr': 'corr-threshold'}
for m, g in order:
    v = agg.get((m, g))
    if not v:
        continue
    h = np.mean(v['h'])
    a = np.mean(v['a']) if v['a'] else None
    L.append(f"{m} & {srcname[g]} & {h:.2f} & "
             + (f"{a:.2f}" if a is not None else '--') + BSB + BSB)
# Grad-AE
ge = [r['hit3'] for r in recs if r['method'] == 'Grad-AE']
if ge:
    L.append(f"Grad (AE scorer) & none & {np.mean(ge):.2f} & --" + BSB + BSB)
L += [BSB + 'bottomrule', BSB + 'end{tabular}', BSB + 'end{table}']
open(os.path.join(TAB, 'tab_e8.tex'), 'w', encoding='utf-8').write(
    chr(10).join(L))
print('-> tab_e8.tex')
