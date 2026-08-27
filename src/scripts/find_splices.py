# -*- coding: utf-8 -*-
"""Find comma splices introduced by the dash conversion:
comma + newline? + clause pronoun starter."""
import re

TEX = r'D:\0科研\工作1\第11篇SCI\paper\DKE_StabilityNotValidity.tex'
t = open(TEX, encoding='utf-8').read()

pat = re.compile(r',\s*\n?\s*(it|this|these|those|we|they|there)\s+'
                 r'([a-z]+)', re.I)
hits = []
for m in pat.finditer(t):
    ctx = t[max(0, m.start() - 60):m.end() + 50].replace('\n', ' ')
    hits.append((m.start(), ctx))
print(f'{len(hits)} candidate(s):')
for _, c in hits:
    print(' *', c)
