# -*- coding: utf-8 -*-
"""修复三处 heredoc 控制字符/断行损伤 + 广谱断裂扫描."""
import re

P = r'D:\0科研\工作1\第11篇SCI\paper\INS_StabilityNotValidity.tex'
t = open(P, encoding='utf-8').read()

n1 = t.count('\x0c\\finding{A published')
t = t.replace('\x0c\\finding{A published', '\\finding{A published')
n2 = t.count('$\x09o$')
t = t.replace('$\x09o$', '$\\to$')
n3 = t.count('Proposition~\nef{prop:cert}')
t = t.replace('Proposition~\nef{prop:cert}', 'Proposition~\\ref{prop:cert}')
open(P, 'w', encoding='utf-8').write(t)
print(f'fix1 formfeed={n1}, fix2 tab-to={n2}, fix3 broken-ref={n3}')

# 广谱扫描: 行尾 ~ 或 $ 后下一行以 1-2 个小写字母{ 开头 (疑似断裂宏)
lines = t.split('\n')
sus = []
for i in range(len(lines) - 1):
    if re.search(r'[~$\\]$|\($', lines[i]) and re.match(r'^[a-z]{1,2}\{', lines[i + 1]):
        sus.append((i + 1, lines[i][-25:], lines[i + 1][:25]))
for s in sus:
    print('疑似断裂:', s)
print('done')
