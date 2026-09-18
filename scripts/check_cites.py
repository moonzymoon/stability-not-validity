# -*- coding: utf-8 -*-
"""检查 bib 条目与正文引用的对应关系。"""
import re

t = open('paper/INS_StabilityNotValidity.tex', encoding='utf-8').read()
bib = open('paper/references.bib', encoding='utf-8').read()

cited = set()
for m in re.findall(r'\\cite[tp]?\{([^}]+)\}', t):
    for k in m.split(','):
        cited.add(k.strip())

bibkeys = re.findall(r'@\w+\{([^,]+),', bib)
counts = {k: len(re.findall(r'\\cite[tp]?\{[^}]*\b' + re.escape(k) + r'\b', t)) for k in bibkeys}

print(f'bib 条目数: {len(bibkeys)}, 正文引用的不同 key 数: {len(cited)}')
print('未被引用的 bib 条目:', [k for k in bibkeys if k not in cited])
print('引用了但 bib 缺失:', sorted(k for k in cited if k not in bibkeys))
print()
print('重点条目引用次数:')
for k in ['ours10', 'su2019smd', 'su2019omnianomaly', 'mishra2026condattr',
          'ismail2020benchmark', 'pedregosa2011sklearn', 'benavoli2014time',
          'runge2019inferring', 'barata2026survey', 'ko2025pbear']:
    print(f'  {k}: {counts.get(k, 0)}')
print()
low = [f'{k}({v})' for k, v in counts.items() if v <= 1]
print('只被引 0-1 次的条目(适配性抽查对象):', low)
