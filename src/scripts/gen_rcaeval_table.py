# -*- coding: utf-8 -*-
"""生成 RCAEval 第四测试床表 (tab_rcaeval.tex) — 从 e31_rcaeval.json."""
import collections
import json

import numpy as np

CACHE = r'D:\0科研\工作1\第11篇SCI\src\_cache'
OUT = r'D:\0科研\工作1\第11篇SCI\paper\tables\tab_rcaeval.tex'

d = json.load(open(f'{CACHE}\\e31_rcaeval.json', encoding='utf-8'))
agg = collections.defaultdict(list)
for r in d:
    agg[(r['method'], r['graph_source'])].append((r['hit1'], r['hit3']))

# (显示名, json键, 图源显示)
rows = [
    ('AERec', 'AERec', '--'),
    ('zDev', 'zDev', '--'),
    ('Random', 'Random', '--'),
    ('DPTA-G', 'DPTA-G', 'clean'),
    ('DPTA-G', 'DPTA-G', 'anom'),
    ('PropRank', 'PropRank', 'clean'),
    ('PropRank', 'PropRank', 'anom'),
    ('GRAA($p{=}0.2$)', 'GRAA(p=0.2)', 'clean'),
    ('GRAA($p{=}0.2$)', 'GRAA(p=0.2)', 'anom'),
    ('GRAA($p{=}0.4$)', 'GRAA(p=0.4)', 'clean'),
    ('GRAA($p{=}0.4$)', 'GRAA(p=0.4)', 'anom'),
]
SRCDISPLAY = {'--': '--', 'clean': 'pcmci\\_clean', 'anom': 'pcmci\\_anom'}

lines = [
    '\\begin{table}[t]\\centering\\small',
    '\\caption{Fourth testbed: RCAEval RE1 Online Boutique --- real'
    ' distributed-system telemetry (46 metrics, 14 services) with'
    ' injected CPU/MEM faults at five services, three repetitions'
    ' (30 units; constructed ground truth: the faulted service\'s'
    ' metric set; hit@$K$ mean; the Random control\'s 0.243 matches'
    ' the 0.244 combinatorial expectation). Resource faults are'
    ' deviational: reconstruction and deviation readers dominate,'
    ' exactly as the deviational strand predicts; source swap costs'
    ' the transmitting method 6.0 points while GRAA($p{=}0.4$) is'
    ' flat ($+1.7$) --- graceful degradation replicates outside'
    ' process control.}',
    '\\label{tab:rcaeval}',
    '\\begin{tabular}{llcc}',
    '\\toprule',
    'Method & Graph source & hit@1 & hit@3\\\\',
    '\\midrule',
]
for disp, key, src in rows:
    src_key = 'none' if src == '--' else f'pcmci_{src}'
    v = agg[(key, src_key)]
    lines.append(f'{disp} & {SRCDISPLAY[src]} & '
                 f'{np.mean([a for a, _ in v]):.2f} & '
                 f'{np.mean([b for _, b in v]):.2f}\\\\')
lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines))
print('wrote', OUT)
