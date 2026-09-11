# -*- coding: utf-8 -*-
"""Fix the 4 comma splices found by find_splices.py."""
TEX = r'D:\0科研\工作1\第11篇SCI\paper\DKE_StabilityNotValidity.tex'
t = open(TEX, encoding='utf-8').read()

REPL = [
    ("confidences is a first-order gate, it can be dominated by a few",
     "confidences is a first-order gate; it can be dominated by a few"),
    ("strong, we conjecture that strong anomalies inflate the residual",
     "strong; we conjecture that strong anomalies inflate the residual"),
    ("smearing the ablation\nsignal across candidates; a hypothesis we flag as such, not a verified\nmechanism)",
     "smearing the ablation\nsignal across candidates, a hypothesis we flag as such rather than a\nverified mechanism)"),
    ("\\textbf{true} graph GRAA loses only 1.1pp (0.609 vs.\\ 0.620), we detect no statistically significant cost",
     "\\textbf{true} graph GRAA loses only 1.1pp (0.609 vs.\\ 0.620); we detect no statistically significant cost"),
    ("future test of the relational strand on distributed systems, we state this scope openly.",
     "future test of the relational strand on distributed systems; we state this scope openly."),
]
ok = 0
for old, new in REPL:
    if old in t:
        t = t.replace(old, new, 1)
        ok += 1
    else:
        print('NOT FOUND:', old[:60].replace('\n', ' '))
open(TEX, 'w', encoding='utf-8').write(t)
print(f'{ok}/5 replaced')
