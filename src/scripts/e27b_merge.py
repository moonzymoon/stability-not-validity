# -*- coding: utf-8 -*-
"""e27b: 合并 e1 粗网格 (mag 0.5) + e27 细网格 -> e27_merged.json (ACR + perturbed hit)."""
import json
import collections

import numpy as np

CACHE = r'D:\0科研\工作1\第11篇SCI\src\_cache'

d = json.load(open(f'{CACHE}\\e1_anchor.json', encoding='utf-8'))
old = collections.defaultdict(list)
for r in d:
    if (r.get('family') == 'del' and r['scorer'] == 'iforest'
            and r['method'] in ('PropRank', 'DPTA-G', 'GraphGranger')
            and r['dataset'].endswith('_m0.5')):
        old[(r['method'], r['graph_source'], r['strength'])].append(r['acr3_base'])

# 扰动后 hit 用 e2 (hit3_pert); e1 的 hit3 是未扰动基线, 不能用
e2 = json.load(open(f'{CACHE}\\e2_scm_quality.json', encoding='utf-8'))
oldhit = collections.defaultdict(list)
for r in e2:
    if (r.get('family') == 'del' and r['scorer'] == 'iforest'
            and r['train_frac'] == 1.0 and r['tag'] == 'scm'
            and r['method'] in ('PropRank', 'DPTA-G', 'GraphGranger')):
        oldhit[(r['method'], r['graph_source'], r['strength'])].append(r['hit3_pert'])

new = json.load(open(f'{CACHE}\\e27_knee_fine.json', encoding='utf-8'))
newd = collections.defaultdict(list)
newhit = collections.defaultdict(list)
for r in new:
    newd[(r['method'], r['graph_source'], r['strength'])].append(r['acr3'])
    newhit[(r['method'], r['graph_source'], r['strength'])].append(r['hit3'])

out = {}
for m in ('PropRank', 'DPTA-G', 'GraphGranger'):
    for g in ('true', 'pcmci_anom'):
        acr, hit = {}, {}
        for src, dst in ((old, acr), (newd, acr), (oldhit, hit), (newhit, hit)):
            for (mm, gg, s), v in src.items():
                if (mm, gg) == (m, g) and s > 0:
                    dst[s] = float(np.mean(v))
        xs = sorted(acr)
        out[f'{m}|{g}'] = dict(
            xs=xs,
            acr=[acr[x] for x in xs],
            hit=[hit[x] for x in xs])
        print(f'{m:13s} {g:11s} eps:', ' '.join(f'{x:.2f}' for x in xs))
        print(f'{"":27s} ACR:', ' '.join(f'{a:.3f}' for a in out[f"{m}|{g}"]["acr"]))
        print(f'{"":27s} hit:', ' '.join(f'{a:.3f}' for a in out[f"{m}|{g}"]["hit"]))

json.dump(out, open(f'{CACHE}\\e27_merged.json', 'w'), indent=1)
print('saved e27_merged.json')
