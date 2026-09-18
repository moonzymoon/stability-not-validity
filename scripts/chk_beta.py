# -*- coding: utf-8 -*-
"""检查 \beta 的真实用法 (避开 shell 转义地狱)."""
import re

t = open(r'D:\0科研\工作1\第11篇SCI\paper\INS_StabilityNotValidity.tex',
         encoding='utf-8').read()
pat = re.compile(r'\\beta')
hits = [m.start() for m in pat.finditer(t)]
print('true beta count:', len(hits))
for h in hits[:10]:
    print(repr(t[h-30:h+20]))
# 其他符号占用检查
for sym in (r'\alpha', r'\theta', r'\rho'):
    print(sym, 'count:', len(re.findall(sym, t)))
