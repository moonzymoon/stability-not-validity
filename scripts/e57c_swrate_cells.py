# -*- coding: utf-8 -*-
"""E57c: SW-rate 的阈值敏感性 — 与论文同口径 (e1_anchor 单元格, K=3).
canonical: ACR>0.8 & hit<0.4, 2700 单元格, 25%/21.7% 头条数字.
敏感性: ts∈{0.7,0.8,0.9} × tv∈{0.3,0.4,0.5}, 全部/去Random,
canonical 处 bootstrap 95% CI (B=2000, 单元格重采样).
输出: _cache/e57c_swrate.json
"""
import json
import os
import collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.dirname(HERE) + '/_cache'

recs = json.load(open(os.path.join(CACHE, 'e1_anchor.json'),
                      encoding='utf-8'))
cells = collections.defaultdict(list)
for r in recs:
    if r.get('method', '').startswith('_') or r.get('family') == 'none':
        continue
    if r.get('acr3_base') is None:
        continue
    cells[(r['dataset'], r['scorer'], r['method'], r['family'],
           r['strength'], r['graph_source'])].append(r)

pts = []
for v in cells.values():
    a = float(np.mean([x['acr3_base'] for x in v]))
    h = float(np.mean([x['hit3'] for x in v]))
    m = v[0]['method']
    pts.append((a, h, m))
print(f'单元格数: {len(pts)}')

out = {'n_cells': len(pts), 'matrix': {}}
A = np.array([p[0] for p in pts])
H = np.array([p[1] for p in pts])
Rnd = np.array([p[2] == 'Random' for p in pts])

for ts in (0.7, 0.8, 0.9):
    for tv in (0.3, 0.4, 0.5):
        sw = (A > ts) & (H < tv)
        out['matrix'][f'ts{ts}_tv{tv}'] = dict(
            all=float(sw.mean()),
            excl_random=float(sw[~Rnd].mean()),
            n_sw=int(sw.sum()))

rng = np.random.default_rng(0)
n = len(pts)
boots = []
for _ in range(2000):
    idx = rng.integers(0, n, n)
    sw = (A[idx] > 0.8) & (H[idx] < 0.4)
    boots.append(sw.mean())
out['canonical_ci'] = dict(ts=0.8, tv=0.4, rate=float(((A > .8) & (H < .4)).mean()),
                           lo=float(np.quantile(boots, 0.025)),
                           hi=float(np.quantile(boots, 0.975)))
json.dump(out, open(os.path.join(CACHE, 'e57c_swrate.json'), 'w'),
          default=float)
for k, v in out['matrix'].items():
    print(k, f"all={v['all']:.3f} exclRnd={v['excl_random']:.3f} n={v['n_sw']}")
print('canonical:', out['canonical_ci'])
