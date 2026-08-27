# -*- coding: utf-8 -*-
TEX = r'D:\0科研\工作1\第11篇SCI\paper\DKE_StabilityNotValidity.tex'
t = open(TEX, encoding='utf-8').read()
REPL = [
    ("certificate holds and, crucially, how often certified windows are still",
     "certificate holds and how often certified windows are still"),
    ("still wrong 36\\% (Wilson CI [24, 50]). Notably, the certificate is",
     "still wrong 36\\% (Wilson CI [24, 50]). The certificate is"),
    ("uncovered as a positive result, our probe is a floor, not a ceiling.",
     "uncovered as a positive result; our probe provides a lower bound, not an upper bound."),
]
for old, new in REPL:
    if old in t:
        t = t.replace(old, new, 1)
        print('ok:', old[:45])
    else:
        print('MISS:', old[:45])
open(TEX, 'w', encoding='utf-8').write(t)
