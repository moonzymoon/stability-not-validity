# -*- coding: utf-8 -*-
"""生成两个附录表: tab_app_scorer (打分器预算分层) + tab_app_gnn (GNN vs 探针)."""
import json
import collections
import os

import numpy as np

CACHE = r'D:\0科研\工作1\第11篇SCI\src\_cache'
TAB = r'D:\0科研\工作1\第11篇SCI\paper\tables'

# ---- scorer budget 表 ----
data = {}
for f, tag in (('e2_scm_quality.json', 'scm'), ('e2_te.json', 'te')):
    d = json.load(open(os.path.join(CACHE, f)))
    ch = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in d:
        if r.get('family') == 'none' and r['scorer'] == 'pca':
            ch[r['method']][r['train_frac']].append(r['hit3'])
    for m, fr in ch.items():
        data[(m, tag)] = {k: float(np.mean(v)) for k, v in fr.items()}

order = ['Grad', 'GlobalCF', 'CondAttr', 'AERec', 'zDev', 'PropRank',
         'DPTA-G', 'GraphGranger', 'Random']
lines = [
    '\\begin{table}[t]\\centering\\small',
    '\\caption{Scorer training-budget stratification (PCA scorer, '
    'hit@3 at 25\\% vs.\\ 100\\% of the normal pool; the anomalous-window '
    'set is held fixed). The gradient reader moves $+8.9$ points on SCM '
    'but $-16.3$ on TE over the same budget sweep --- the effect is '
    'dataset-dependent --- while graph-consuming and deviation readers '
    'are budget-flat and the remaining graph-free readers move by at '
    'most 3.6 points (GlobalCF, TE).}',
    '\\label{tab:app-scorer}',
    '\\begin{tabular}{lcccc}',
    '\\toprule',
    ' & \\multicolumn{2}{c}{SCM} & \\multicolumn{2}{c}{TE}\\\\',
    '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}',
    'Method & 25\\% & 100\\% & 25\\% & 100\\%\\\\',
    '\\midrule',
]
for m in order:
    if (m, 'scm') not in data:
        continue
    s, t = data[(m, 'scm')], data[(m, 'te')]
    lines.append(f"{m} & {s[0.25]:.2f} & {s[1.0]:.2f} & "
                 f"{t[0.25]:.2f} & {t[1.0]:.2f}\\\\")
lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
open(os.path.join(TAB, 'tab_app_scorer.tex'), 'w',
     encoding='utf-8').write('\n'.join(lines))
print('wrote tab_app_scorer.tex')

# ---- GNN vs probe 表 ----
d = json.load(open(os.path.join(CACHE, 'e17_gnn_positive.json')))
agg = collections.defaultdict(list)
for r in d:
    agg[r['graph_source']].append(r['hit3'])
gnn = {k: float(np.mean(v)) for k, v in agg.items()}
lines = [
    '\\begin{table}[t]\\centering\\small',
    '\\caption{Trained GNN ranker vs.\\ parameter-free GCN probe on '
    'identical conditions (hit@3, SCM medium). Even with the '
    'cross-distribution barrier removed (within-seed '
    'leave-one-episode-out training), the trained ranker reaches only '
    '0.27--0.31 against the probe\\textquotesingle s 0.42--0.49, ruling '
    'out data scarcity as the sole explanation of its poor transfer '
    '(cross-seed: 0.00--0.31, three designs; details in the released '
    'cache).}',
    '\\label{tab:app-gnn}',
    '\\begin{tabular}{llcc}',
    '\\toprule',
    'Ranker & Training & true graph & learned graph\\\\',
    '\\midrule',
    f"GNN (trained) & within-seed LOEO & {gnn.get('true', 0):.2f} & "
    f"{gnn.get('pcmci', 0):.2f}\\\\",
    'GCN probe & parameter-free & 0.49 & 0.42\\\\',
    '\\bottomrule',
    '\\end{tabular}',
    '\\end{table}',
    '',
]
open(os.path.join(TAB, 'tab_app_gnn.tex'), 'w',
     encoding='utf-8').write('\n'.join(lines))
print('wrote tab_app_gnn.tex')
