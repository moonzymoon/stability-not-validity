# -*- coding: utf-8 -*-
TEX = r'D:\0科研\工作1\第11篇SCI\paper\DKE_StabilityNotValidity.tex'
t = open(TEX, encoding='utf-8').read()
old = "\\author[a]{Yao Zhang}\n"
new = "\\author[a]{Yao Zhang}\\corref{cor1}\n"
if old in t and '\\corref{cor1}' not in t:
    t = t.replace(old, new, 1)
    print('corref added')
else:
    print('skip:', '\\corref{cor1}' in t, old in t)
open(TEX, 'w', encoding='utf-8').write(t)
