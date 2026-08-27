# -*- coding: utf-8 -*-
"""生成 GRAA p×gamma 敏感性附表 (tab_graa_sens.tex)."""
import json

import numpy as np
import collections

CACHE = r'D:\0科研\工作1\第11篇SCI\src\_cache'
OUT = r'D:\0科研\工作1\第11篇SCI\paper\tables\tab_graa_sens.tex'

d = json.load(open(f'{CACHE}\\e28_graa_sens.json', encoding='utf-8'))
agg = collections.defaultdict(list)
for r in d:
    agg[(r['graph_source'], r['p'], r['gamma'])].append(r['hit3'])

PS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
GS = [0.5, 1.0, 2.0]


def cell(g, p, ga, bold=False):
    v = np.mean(agg[(g, p, ga)])
    s = f'{v:.3f}'
    return f'\\textbf{{{s}}}' if bold else s


lines = [
    '\\begin{table}[t]',
    '\\centering',
    '\\small',
    '\\caption{GRAA sensitivity: hit@3 (mean over five SCM seeds) across'
    ' prune fraction $p$ and mixing exponent $\\beta$, under true and'
    ' contaminated graphs. Bold: the deployed default $p{=}0.4$,'
    ' $\\beta{=}1$. Contaminated-graph performance rises monotonically'
    ' in $p$ up to the peak at $p{=}0.4$; true-graph cost stays within'
    ' 1.7pp for $p\\le 0.5$ and collapses only at $p{=}0.6$; $\\beta$'
    ' moves results by at most 2pp --- GRAA has one operative'
    ' parameter.}',
    '\\label{tab:graa-sens}',
    '\\begin{tabular}{l' + 'c' * 6 + '}',
    '\\toprule',
    ' & \\multicolumn{3}{c}{True graph} & \\multicolumn{3}{c}{Contaminated} \\\\',
    '\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}',
    '$p$ & $\\beta{=}0.5$ & $\\beta{=}1$ & $\\beta{=}2$'
    ' & $\\beta{=}0.5$ & $\\beta{=}1$ & $\\beta{=}2$ \\\\',
    '\\midrule',
]
for p in PS:
    bold = abs(p - 0.4) < 1e-9
    row = [f'{p:.1f}']
    for g in ('true', 'pcmci_anom'):
        for ga in GS:
            row.append(cell(g, p, ga, bold=bold and abs(ga - 1.0) < 1e-9))
    lines.append(' & '.join(row) + ' \\\\')
lines += [
    '\\bottomrule',
    '\\end{tabular}',
    '\\end{table}',
    '',
]
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines))
print('wrote', OUT)
