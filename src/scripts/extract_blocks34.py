# -*- coding: utf-8 -*-
"""提取块3(定理/引理/证明)与块4(实验部分文本, 去图) —— 原文逐字, 不改写."""
import re

t = open('paper/DKE_StabilityNotValidity.tex', encoding='utf-8').read()

# 块3: 全部 proposition / corollary / theorem / lemma / proof 环境(逐字)
pat = re.compile(
    r'\\begin\{(proposition|corollary|theorem|lemma)\*?\}.*?'
    r'\\end\{\1\*?\}(?:\s*\\begin\{proof\}.*?\\end\{proof\})?', re.S)
b3 = pat.findall(t)
b3_blocks = [m.group(0) for m in re.finditer(
    r'\\begin\{(proposition|corollary|theorem|lemma)\*?\}.*?'
    r'\\end\{\1\*?\}(?:\s*\\begin\{proof\}.*?\\end\{proof\})?', t, re.S)]

# 块4: Experiments 节(含其全部 subsection), 到 Discussion 节前; 去图
i = t.find('\\section{Experiments}')
j = t.find('\\section{Discussion')
b4 = t[i:j]
b4 = re.sub(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}', '', b4, flags=re.S)

out = []
out.append('=' * 30 + ' 块3：定理/引理/证明 (' + str(len(b3_blocks)) + '个环境) ' + '=' * 30)
for k, b in enumerate(b3_blocks, 1):
    out.append(f'\n----- 块3.{k} -----\n' + b.strip())
out.append('\n' + '=' * 30 + ' 块4：实验部分文本(去图) ' + '=' * 30)
out.append(b4.strip())
open('blocks34.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('块3环境数:', len(b3_blocks))
print('块4字符数:', len(b4))
