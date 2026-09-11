# -*- coding: utf-8 -*-
"""生成难度分层四象限附录表 (tab_app_tier.tex)."""
import json

CACHE = r'D:\0科研\工作1\第11篇SCI\src\_cache'
OUT = r'D:\0科研\工作1\第11篇SCI\paper\tables\tab_app_tier.tex'

a1 = json.load(open(f'{CACHE}\\a1_analysis.json', encoding='utf-8'))
bm = a1['quadrant']['by_mag']
TIER = [('0.25', 'weak'), ('0.5', 'medium'), ('1.5', 'strong')]

lines = [
    '\\begin{table}[t]\\centering\\small',
    '\\caption{Quadrant decomposition stratified by difficulty tier '
    '(SCM grid, 900 cells per tier; thresholds as in '
    'Table~\\ref{tab:quadrant}; rows sum to 1 before rounding). '
    'The stable-wrong share grows as anomalies weaken (0.36/0.24/0.16) '
    '--- the regime where operators need attribution most.}',
    '\\label{tab:app-tier}',
    '\\begin{tabular}{lcccc}',
    '\\toprule Tier & Stable-wrong & Stable-right & Unstable-right & '
    'Unstable-wrong\\\\',
    '\\midrule',
]
for mag, name in TIER:
    q = bm[mag]
    lines.append(f'{name} ({mag}) & {q["stable_wrong"]:.2f} & '
                 f'{q["stable_right"]:.2f} & {q["unstable_right"]:.2f} & '
                 f'{q["unstable_wrong"]:.2f}\\\\')
lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines))
print('wrote', OUT)
