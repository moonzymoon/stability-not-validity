# -*- coding: utf-8 -*-
"""Reduce em-dash density (AI writing tell) in the manuscript.

Rules:
1. Paired parenthetical  " --- X --- "  (X has no period, <=160 chars) -> " (X) "
2. Remaining single " --- " -> "; " if an independent clause follows
   (conjugated verb within first 6 words), else ", "
Only touches spaced ' --- ' outside math. Prints before/after counts and
all transformation contexts for manual review.
"""
import re

TEX = r'D:\0科研\工作1\第11篇SCI\paper\DKE_StabilityNotValidity.tex'
t = open(TEX, encoding='utf-8').read()
n0 = t.count(' --- ') + len(re.findall(r'(?<=[a-zA-Z%\$])---(?=[a-zA-Z])', t))

# ---- rule 1: paired parentheticals (no '.' inside, bounded length) ----
def pair_sub(m):
    inner = m.group(1)
    if '.' in inner or len(inner) > 160 or inner.count('---'):
        return m.group(0)
    return ' (' + inner + ') '
t2 = re.sub(r' --- (.+?) --- ', pair_sub, t)

# ---- rule 2: singles by clause independence ----
VERBS = {'is', 'are', 'was', 'were', 'has', 'have', 'had', 'does', 'do',
         'did', 'can', 'cannot', 'could', 'may', 'might', 'will', 'would',
         'should', 'must', 'remains', 'shows', 'show', 'gives', 'give',
         'yields', 'yield', 'leaves', 'leave', 'requires', 'require',
         'provides', 'provide', 'suggests', 'suggest', 'keeps', 'keep',
         'allows', 'allow', 'makes', 'make', 'means', 'implies', 'carries',
         'consumes', 'degrades', 'resists', 'loses', 'loses', 'gains',
         'reaches', 'fills', 'covers', 'changes', 'breaks', 'holds',
         'transfers', 'collapses', 'helps', 'fails', 'works', 'reads'}

def single_sub(m):
    nxt = m.group(1)
    words = re.findall(r"[A-Za-z\\']+|\$", nxt)[:7]
    flat = [w.strip("\\{}$").lower() for w in words]
    indep = any(w in VERBS for w in flat[:6])
    return '; ' if indep else ', '

log = []
def log_sub(m):
    rep = single_sub(m)
    log.append((m.group(0)[:60].replace('\n', ' '), rep.strip()))
    return rep
t3 = re.sub(r' ---\s*\n?\s*([a-z\\])', log_sub, t2)

n1 = t3.count(' --- ') + len(re.findall(r'(?<=[a-zA-Z%\$])---(?=[a-zA-Z])', t3))
open(TEX, 'w', encoding='utf-8').write(t3)
print(f'spaced/unspaced em-dash tokens: {n0} -> {n1}')
print(f'paired->parens + singles converted: {len(log)} single conversions')
for a, b in log[:15]:
    print(f'  [{a}...] -> {b}')
