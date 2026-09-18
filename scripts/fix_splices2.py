# -*- coding: utf-8 -*-
TEX = r'D:\0科研\工作1\第11篇SCI\paper\INS_StabilityNotValidity.tex'
t = open(TEX, encoding='utf-8').read()
REPL = [
    ("vs.\\ 0.620), we detect no statistically significant cost",
     "vs.\\ 0.620); we detect no statistically significant cost"),
    ("relational strand on distributed systems, we state this scope",
     "relational strand on distributed systems; we state this scope"),
]
for old, new in REPL:
    if old in t:
        t = t.replace(old, new, 1)
        print('ok:', old[:40])
    else:
        print('MISS:', old[:40])
open(TEX, 'w', encoding='utf-8').write(t)
