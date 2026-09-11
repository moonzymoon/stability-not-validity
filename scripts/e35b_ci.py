# -*- coding: utf-8 -*-
"""E35b: 两级窗口 bootstrap CI -> 重生成 tab_main_scm/te 的 CI 列."""
import collections
import json
import os

import numpy as np

CACHE = r'D:\0科研\工作1\第11篇SCI\src\_cache'
TAB = r'D:\0科研\工作1\第11篇SCI\paper\tables'

store = json.load(open(os.path.join(CACHE, 'e35_wlhit.json'), encoding='utf-8'))

# 聚合: (tag, method, source) -> 各 (ds,scorer) cell 的窗口向量
cells = collections.defaultdict(dict)
for k, v in store.items():
    tag, ds, sk, m, src = k.split('|')
    cells[(tag, m, src)][(ds, sk)] = np.array(v['hit3'], float)

rng = np.random.default_rng(0)
B = 2000


def wl_ci(cellmap):
    """两级 bootstrap: cell 内重采样窗口 -> cell 均值 -> 跨 cell 平均."""
    keys = sorted(cellmap)
    arrs = [cellmap[k] for k in keys]
    means = []
    for _ in range(B):
        vals = [a[rng.integers(0, len(a), len(a))].mean() for a in arrs]
        means.append(np.mean(vals))
    lo, hi = np.percentile(means, [2.5, 97.5])
    point = float(np.mean([a.mean() for a in arrs]))
    return point, lo, hi


def fmt(x):
    return f'{x:.3f}'[1:] if 0 <= x < 1 else f'{x:.3f}'


# 验证点估计与现表一致
check = [('scm', 'DPTA-G', 'true'), ('scm', 'PropRank', 'pcmci_anom'),
         ('scm', 'zDev', 'none'), ('te', 'zDev', 'pcmci')]
print('== 点估计核对 (应与现表一致) ==')
results = {}
for tag in ('scm', 'te'):
    for (t, m, src) in [k for k in cells if k[0] == tag]:
        p, lo, hi = wl_ci(cells[(t, m, src)])
        results[(t, m, src)] = (p, lo, hi)
        if (t, m, src) in [tuple(x) for x in check]:
            print(f'  {m}/{src}: point={p:.3f} CI=[{lo:.3f},{hi:.3f}]')

# 生成行文本
scm_rows = [
    ('DPTA-G', 'true'), ('DPTA-G', 'PCMCI-clean'), ('DPTA-G', 'PCMCI-anom'),
    ('PropRank', 'true'), ('PropRank', 'PCMCI-clean'),
    ('PropRank', 'PCMCI-anom'),
    ('GraphGranger', 'true'), ('GraphGranger', 'PCMCI-clean'),
    ('GraphGranger', 'PCMCI-anom'),
    ('GlobalCF', 'none'), ('AERec', 'none'), ('Grad', 'none'),
    ('zDev', 'none'), ('CondAttr', 'none'), ('Random', 'none'),
]
te_rows = [
    ('DPTA-G', 'PCMCI'), ('PropRank', 'PCMCI'), ('GraphGranger', 'PCMCI'),
    ('GlobalCF', 'none'), ('AERec', 'none'), ('Grad', 'none'),
    ('zDev', 'none'), ('CondAttr', 'none'), ('Random', 'none'),
]
src_map = {'PCMCI-clean': 'pcmci_clean', 'PCMCI-anom': 'pcmci_anom',
           'PCMCI': 'pcmci', 'none': 'none', 'true': 'true'}

out = {'scm': {}, 'te': {}}
for tag, rows in (('scm', scm_rows), ('te', te_rows)):
    for m, src in rows:
        key = (tag, m, src_map[src])
        if key not in results:
            print('MISSING', key)
            continue
        p, lo, hi = results[key]
        out[tag][f'{m}|{src}'] = f'{fmt(lo)},{fmt(hi)}'
        print(f'{tag} {m:13s} {src:12s} point={p:.3f} '
              f'WL-CI=[{fmt(lo)},{fmt(hi)}]')

json.dump(out, open(os.path.join(CACHE, 'e35_wlci.json'), 'w'), indent=1)
print('saved e35_wlci.json')
