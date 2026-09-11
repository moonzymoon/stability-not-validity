# -*- coding: utf-8 -*-
"""严审-机械维度扫描: 术语一致性 / 遗留标记 / 常见笔误 / 记号统一."""
import re
import collections

t = open(r'D:\0科研\工作1\第11篇SCI\paper\DKE_StabilityNotValidity.tex',
         encoding='utf-8').read()
issues = []

# 术语大小写/拼写一致性
for term, variants in {
    'zDev': ['Zdev', 'zdev', 'ZDev', 'zDEV'],
    'PropRank': ['Proprank', 'propRank', 'PROPRANK'],
    'DPTA-G': ['DPTA-g', 'dpta-G', 'Dpta-G'],
    'GRAA': ['Graa', 'graa'],
    'RCAEval': ['RCAeval', 'RCAEVAL', 'rcaeval'],
    'PCMCI': ['pcmci '],
    'SWaT': ['Swat ', 'SWAT'],
    'AERec': ['AEREC', 'aerec', 'AeRec'],
}.items():
    for v in variants:
        n = len(re.findall(r'(?<![a-zA-Z\\])' + re.escape(v), t))
        if n:
            issues.append(f'术语不一致: {v!r} x{n} (应为 {term})')

# 遗留标记
for pat in ('TODO', 'FIXME', 'XXX', 'PLACEHOLDER', 'placeholder',
            'lorem', 'TBD'):
    if re.search(pat, t):
        for m in re.finditer(pat, t):
            ln = t[:m.start()].count('\n') + 1
            issues.append(f'遗留标记 {pat} @line {ln}')

# 常见笔误
for w in ('teh ', 'recieve', 'seperate', 'occured', 'comparision',
          'supercede', 'extention'):
    if w in t.lower():
        issues.append(f'拼写: {w.strip()}')

# 记号: hit@K 写法
h3 = len(re.findall(r'hit@3', t))
hK = len(re.findall(r'hit@K', t))
hk = len(re.findall(r'\\hit@K', t))
print(f'hit@3 x{h3}, hit@K x{hK}')
acr = len(re.findall(r'ACR@3', t)) + len(re.findall(r'\\acr@3', t))
print(f'ACR@3-类 x{acr}')

# 图表引用完整性 (0 undefined 已由编译保证; 这里查从未被引用的表/图)
labels = set(re.findall(r'\\label\{(tab|fig|alg|sec):([^}]+)\}', t))
refs = set(re.findall(r'\\ref\{([^}]+)\}', t)) | set(
    re.findall(r'\\cref\{([^}]+)\}', t))
unref = [(k, n) for k, n in labels if f'{k}:{n}' not in refs]
print('未被正文引用的 label:', unref if unref else '无')

print('\n== 问题清单 ==')
print('\n'.join(issues) if issues else '机械维度无问题')
