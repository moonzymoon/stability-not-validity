# -*- coding: utf-8 -*-
"""生成增强实验表格: SWaT 第三测试床 + 增强/边界结果汇总."""
import json
import os
import collections

import numpy as np

CACHE = '_cache'
TAB = os.path.join('..', 'paper', 'tables')
BSB = chr(92)

# ---- SWaT 表 ----
recs = json.load(open(os.path.join(CACHE, 'e10_swat.json'), encoding='utf-8'))
agg = collections.defaultdict(lambda: dict(h1=[], h3=[], h5=[], a=[]))
for r in recs:
    k = (r['method'], r['graph_source'])
    agg[k]['h1'].append(r['hit1'])
    agg[k]['h3'].append(r['hit3'])
    agg[k]['h5'].append(r['hit5'])
    if r.get('acr3') is not None:
        agg[k]['a'].append(r['acr3'])

L = [BSB + 'begin{table}[t]' + BSB + 'centering' + BSB + 'small',
     BSB + 'caption{Third testbed: SWaT (51 sensors, real process data) '
     'with injected variable-level faults --- constructed ground truth on '
     'real data. Validity (hit@$K$, mean over 2 scorers) and stability '
     '(ACR@3) by graph source: silver-reference graph (19 documented '
     'edges), PCMCI-learned, correlation-threshold. Source transmission '
     'and pseudo-stability replicate on real data with healthy absolute '
     'numbers (contrast the TE stress test).}',
     BSB + 'label{tab:swat}',
     BSB + 'begin{tabular}{llcccc}', BSB + 'toprule',
     'Method & Source & hit@1 & hit@3 & hit@5 & ACR@3' + BSB + BSB,
     BSB + 'midrule']
order = [('Grad', 'none'), ('AERec', 'none'), ('DPTA-G', 'silver'),
         ('DPTA-G', 'pcmci'), ('DPTA-G', 'corr'), ('zDev', 'none'),
         ('PropRank', 'silver'), ('PropRank', 'pcmci'), ('PropRank', 'corr'),
         ('CondAttr', 'none'), ('GlobalCF', 'none'),
         ('GraphGranger', 'pcmci'), ('GCN-Rank', 'silver'), ('Random', 'none')]
srcname = {'silver': 'silver', 'pcmci': 'PCMCI', 'corr': 'corr', 'none': '--'}
for m, g in order:
    v = agg.get((m, g))
    if not v:
        continue
    h1 = np.mean(v['h1']); h3 = np.mean(v['h3']); h5 = np.mean(v['h5'])
    a = np.mean(v['a']) if v['a'] else None
    L.append(f"{m} & {srcname[g]} & {h1:.2f} & {h3:.2f} & {h5:.2f} & "
             + (f"{a:.2f}" if a is not None else '--') + BSB + BSB)
L += [BSB + 'bottomrule', BSB + 'end{tabular}', BSB + 'end{table}']
open(os.path.join(TAB, 'tab_swat.tex'), 'w', encoding='utf-8').write(
    chr(10).join(L))
print('-> tab_swat.tex')

# ---- 增强与边界结果表 (P1/P2/P3/P4/P8c/P8d) ----
p2 = json.load(open(os.path.join(CACHE, 'e11_p2.json')))
p2agg = collections.defaultdict(lambda: dict(g=[], r=[], b=[], o=[]))
for r in p2:
    a = p2agg[(r['method'], r['k'])]
    a['g'].append(r['hit_greedy']); a['r'].append(r['hit_random'])
    a['b'].append(r['hit_base']); a['o'].append(r['hit_oracle'])
p3 = json.load(open(os.path.join(CACHE, 'e11_p3.json')))
p3agg = collections.defaultdict(list)
for r in p3:
    p3agg[(r['method'], r['certified'])].append(r['hit'])
p8c = json.load(open(os.path.join(CACHE, 'e12_p8c.json')))

L = [BSB + 'begin{table}[t]' + BSB + 'centering' + BSB + 'footnotesize',
     BSB + 'caption{Enhancement and boundary results (SCM medium unless '
     'stated). (a) Active edge verification under systematic bias: '
     'greedy (influence-ranked) vs random edge deletion, PropRank. '
     '(b) Stability certificates: validity of windows holding a '
     'perturbation-robustness certificate vs the rest. (c) Per-method '
     'reliability predictors (LODO). (d) Consensus-set size vs validity '
     '(PropRank, q=0.8).}',
     BSB + 'label{tab:boost}',
     BSB + 'begin{tabular}{lll}', BSB + 'toprule',
     'Panel & Quantity & Value' + BSB + BSB, BSB + 'midrule']
row = p2agg[('PropRank', 5)]
L.append(f"(a) & PropRank verify k=5: base $\\to$ greedy / random & "
         f"{np.mean(row['b']):.2f} $\\to$ {np.mean(row['g']):.2f} / "
         f"{np.mean(row['r']):.2f}" + BSB + BSB)
row = p2agg[('PropRank', 12)]
L.append(f"(a) & k=12 & {np.mean(row['b']):.2f} $\\to$ "
         f"{np.mean(row['g']):.2f} / {np.mean(row['r']):.2f} "
         f"(oracle {np.mean(row['o']):.2f})" + BSB + BSB)
ct = p3agg[('DPTA-G', True)]; cf = p3agg[('DPTA-G', False)]
L.append(f"(b) & DPTA-G certified (envelope rule): n, hit@3 & {len(ct)}, "
         f"{np.mean(ct):.2f}" + BSB + BSB)
L.append(f"(b) & uncertified windows: n, hit@3 & {len(cf)}, "
         f"{np.mean(cf):.2f}" + BSB + BSB)
L.append("(c) & per-method LODO AUROC (DPTA-G/GG/PropRank) & "
         f"{p8c['DPTA-G']:.2f} / {p8c['GraphGranger']:.2f} / "
         f"{p8c['PropRank']:.2f}" + BSB + BSB)
L.append("(d) & consensus set size 0/1/2/3 $\\to$ hit@3 (n) & "
         "0.00/0.27/0.55/0.74 (27/142/158/23)" + BSB + BSB)
L += [BSB + 'bottomrule', BSB + 'end{tabular}', BSB + 'end{table}']
open(os.path.join(TAB, 'tab_boost.tex'), 'w', encoding='utf-8').write(
    chr(10).join(L))
print('-> tab_boost.tex')
