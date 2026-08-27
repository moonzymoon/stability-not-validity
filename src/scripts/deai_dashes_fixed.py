# -*- coding: utf-8 -*-
"""Corrected combined em-dash pass (restores from backup first).

Pass-1 bug ate the first letter after each single-dash conversion;
this version re-emits the captured letter."""
import re

TEX = r'D:\0科研\工作1\第11篇SCI\paper\DKE_StabilityNotValidity.tex'
BAK = TEX.replace('DKE_StabilityNotValidity.tex', '.tex_backup_predash')
t = open(BAK, encoding='utf-8').read()

VERBS = {'is', 'are', 'was', 'were', 'has', 'have', 'had', 'does', 'do',
         'did', 'can', 'cannot', 'could', 'may', 'might', 'will', 'would',
         'should', 'must', 'remains', 'shows', 'show', 'gives', 'give',
         'yields', 'yield', 'leaves', 'leave', 'requires', 'require',
         'provides', 'provide', 'suggests', 'suggest', 'keeps', 'keep',
         'allows', 'allow', 'makes', 'make', 'means', 'implies', 'carries',
         'consumes', 'degrades', 'resists', 'loses', 'gains', 'reaches',
         'fills', 'covers', 'changes', 'breaks', 'holds', 'transfers',
         'collapses', 'helps', 'fails', 'works', 'reads', 'admits',
         'realizes', 'reflects'}

def clause_next(nxt):
    words = re.findall(r"[A-Za-z\\']+|\$", nxt)[:8]
    flat = [w.strip("\\{}$").lower() for w in words]
    return any(w in VERBS for w in flat[:6])

# 1) paired parentheticals -> (...)
def pair_sub(m):
    inner = m.group(1)
    if '.' in inner or len(inner) > 160 or '---' in inner:
        return m.group(0)
    return ' (' + inner + ') '
t = re.sub(r' --- (.+?) --- ', pair_sub, t)

# 2) single ' --- X' where X starts lowercase letter or backslash: keep X!
def single_sub(m):
    rep = '; ' if clause_next(m.group(0)[5:65]) else ', '
    return rep + m.group(1)
t = re.sub(r' ---\s*\n?\s*([a-z\\])', single_sub, t)

# 3) remaining ' --- X' with uppercase/$/digit start
out, pos = [], 0
while True:
    i = t.find(' --- ', pos)
    if i < 0:
        break
    nxt = t[i + 5:i + 65]
    rep = '; ' if clause_next(nxt) else ', '
    out.append(t[pos:i] + rep)
    pos = i + 5
out.append(t[pos:])
t = ''.join(out)

# 4) unspaced word---word
t = re.sub(r'([a-zA-Z])---([a-zA-Z])', r'\1, \2', t)

open(TEX, 'w', encoding='utf-8').write(t)
print('remaining --- (line-start leftovers + ranges):', t.count('---'))
for ln in t.split('\n'):
    if '---' in ln:
        print(' |', ln.strip()[:110])
