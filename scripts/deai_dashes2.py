# -*- coding: utf-8 -*-
"""Second pass: remaining em-dashes after deai_dashes.py (uppercase/$/digit
starts were skipped). Same clause heuristic; unspaced word---word -> ', '."""
import re

TEX = r'D:\0科研\工作1\第11篇SCI\paper\INS_StabilityNotValidity.tex'
t = open(TEX, encoding='utf-8').read()

VERBS = {'is', 'are', 'was', 'were', 'has', 'have', 'had', 'does', 'do',
         'did', 'can', 'cannot', 'could', 'may', 'might', 'will', 'would',
         'should', 'must', 'remains', 'shows', 'show', 'gives', 'give',
         'yields', 'yield', 'leaves', 'leave', 'requires', 'require',
         'provides', 'provide', 'suggests', 'suggest', 'keeps', 'keep',
         'allows', 'allow', 'makes', 'make', 'means', 'implies', 'carries',
         'consumes', 'degrades', 'resists', 'loses', 'gains', 'reaches',
         'fills', 'covers', 'changes', 'breaks', 'holds', 'transfers',
         'collapses', 'helps', 'fails', 'works', 'reads', 'admits',
         'realizes', 'reflects', 'driven', 'rather'}

def clause_next(nxt):
    words = re.findall(r"[A-Za-z\\']+|\$", nxt)[:8]
    flat = [w.strip("\\{}$").lower() for w in words]
    return any(w in VERBS for w in flat[:6])

out = []
pos = 0
res = []
while True:
    i = t.find(' --- ', pos)
    if i < 0:
        break
    nxt = t[i + 5:i + 60]
    rep = '; ' if clause_next(nxt) else ', '
    out.append(t[pos:i] + rep)
    pos = i + 5
out.append(t[pos:])
t = ''.join(out)

# unspaced letters: word---word  (but not '--' ranges / '---' line breaks)
t = re.sub(r'([a-zA-Z])---([a-zA-Z])', r'\1, \2', t)

open(TEX, 'w', encoding='utf-8').write(t)
import subprocess
n = t.count('---')
print('remaining --- occurrences (incl. legit page ranges etc.):', n)
for ln in t.split('\n'):
    if '---' in ln:
        print(' |', ln.strip()[:110])
