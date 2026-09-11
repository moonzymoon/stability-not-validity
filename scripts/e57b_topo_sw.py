# -*- coding: utf-8 -*-
"""E57b: C(拓扑缩放定律拟合) + D-part1(SW-rate 阈值敏感性).
C: e48_topo.json 54条 {d, seed, graph, method, acr3, hit3, stable_wrong}
   -> 按 (d, method) 聚合 stable-wrong 率与 acr3/hit3; 拟合 swrate ~ a + b*d 及
      acr3 ~ a + b*d; 报 R^2.
D-SW: e8_grid.json 95条 {acr3, hit3, ...}
   -> SW-rate(τs, τv) = mean[acr3>=τs & hit3<τv] 的阈值敏感性矩阵;
      并找出复现论文 21.7%(去Random)/25.2%(含) 的 (τs, τv).
输出: _cache/e57b_topo_sw.json
"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.dirname(HERE) + '/_cache'


def linfit(x, y):
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * np.array(x)
    ss = 1 - np.sum((np.array(y) - yhat) ** 2) / max(
        np.sum((np.array(y) - np.mean(y)) ** 2), 1e-12)
    return dict(intercept=float(a), slope=float(b), r2=float(ss))


rec = [r for r in json.load(open(os.path.join(CACHE, 'e48_topo.json'),
                                encoding='utf-8'))
       if 'stable_wrong' in r]  # GRAA 行无 ACR, 剔除
C = {'per_cell': {}, 'fits': {}}
for m in sorted(set(r['method'] for r in rec)):
    cells = {}
    for d in sorted(set(r['d'] for r in rec)):
        sub = [r for r in rec if r['method'] == m and r['d'] == d]
        cells[d] = dict(n=len(sub),
                        sw=float(np.mean([r['stable_wrong'] for r in sub])),
                        acr3=float(np.mean([r['acr3'] for r in sub])),
                        hit3=float(np.mean([r['hit3'] for r in sub])))
    C['per_cell'][m] = cells
    ds = sorted(cells)
    C['fits'][m] = dict(
        sw_vs_d=linfit(ds, [cells[d]['sw'] for d in ds]),
        acr_vs_d=linfit(ds, [cells[d]['acr3'] for d in ds]),
        hit_vs_d=linfit(ds, [cells[d]['hit3'] for d in ds]))

grid_all = json.load(open(os.path.join(CACHE, 'e8_grid.json'),
                          encoding='utf-8'))
grid = [r for r in grid_all if r.get('acr3') is not None]
print(f'e8_grid: {len(grid)}/{len(grid_all)} 条有acr3')
SW = {'matrix': {}, 'headline_search': []}
for ts in (0.7, 0.8, 0.9):
    for tv in (0.3, 0.5):
        both = np.mean([(r['acr3'] >= ts and r['hit3'] < tv) for r in grid])
        noRnd = [r for r in grid if r['method'] != 'Random']
        wo = np.mean([(r['acr3'] >= ts and r['hit3'] < tv) for r in noRnd])
        SW['matrix'][f'ts{ts}_tv{tv}'] = dict(all=float(both),
                                              excl_random=float(wo))
        SW['headline_search'].append((f'ts{ts}_tv{tv}',
                                      round(float(both), 3),
                                      round(float(wo), 3)))

out = dict(C_topology=C, D_swrate=SW)
json.dump(out, open(os.path.join(CACHE, 'e57b_topo_sw.json'), 'w'),
          default=float)

print('=== C: 拓扑缩放 (per d) ===')
for m, cells in C['per_cell'].items():
    for d, c in cells.items():
        print(f'{m} d={d}: sw={c["sw"]:.3f} acr3={c["acr3"]:.3f} hit3={c["hit3"]:.3f} (n={c["n"]})')
    f = C['fits'][m]
    print(f'  拟合: sw~d 斜率{f["sw_vs_d"]["slope"]:.4f} R2={f["sw_vs_d"]["r2"]:.3f} | '
          f'acr~d 斜率{f["acr_vs_d"]["slope"]:.4f} R2={f["acr_vs_d"]["r2"]:.3f} | '
          f'hit~d R2={f["hit_vs_d"]["r2"]:.3f}')
print()
print('=== D: SW-rate 阈值敏感性 (all / 去Random) ===')
for k, v in SW['matrix'].items():
    print(f'{k}: {v["all"]:.3f} / {v["excl_random"]:.3f}')
