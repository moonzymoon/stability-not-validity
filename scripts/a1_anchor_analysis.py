# -*- coding: utf-8 -*-
"""E1 锚点深度分析: 象限分解 / 膝盖 / 双参照 / Go-No-Go 判定。
输出: _cache/a1_analysis.json + 控制台报告
"""
import os
import sys
import json
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')
GRAPH_DEP = {'DPTA-G', 'PropRank', 'GraphGranger'}


def load():
    with open(os.path.join(CACHE, 'e1_anchor.json'), encoding='utf-8') as f:
        return json.load(f)


def main():
    recs = load()
    # cell 聚合: (ds, scorer, method, family, strength, source) -> list[(acr, hit3)]
    cells = collections.defaultdict(list)
    hit_cells = collections.defaultdict(list)   # family='none' 的 hit
    for r in recs:
        if r.get('method', '').startswith('_'):
            continue
        key = (r['dataset'], r['scorer'], r['method'], r['family'],
               r['strength'], r['graph_source'])
        if r['family'] == 'none':
            hit_cells[(r['dataset'], r['scorer'], r['method'],
                       r['graph_source'])].append(
                (r['hit1'], r['hit3'], r['hit5'], r['rbo']))
            continue
        if r.get('acr3_base') is not None:
            cells[key].append((r['acr3_base'], r['hit3']))

    out = {}

    # ---------- 1) Go/No-Go: 象限 ----------
    pts = [(np.mean([a for a, _ in v]), np.mean([h for _, h in v]))
           for v in cells.values()]
    q_all = M.quadrant_stats(pts)
    # 方法×象限分解
    by_method = collections.defaultdict(list)
    for (ds, sc, m, fam, st, src), v in cells.items():
        by_method[m].append((np.mean([a for a, _ in v]),
                             np.mean([h for _, h in v])))
    q_method = {m: M.quadrant_stats(v) for m, v in by_method.items()}
    # 幅度×象限
    by_mag = collections.defaultdict(list)
    for (ds, sc, m, fam, st, src), v in cells.items():
        mag = ds.split('_m')[-1]
        by_mag[mag].append((np.mean([a for a, _ in v]),
                            np.mean([h for _, h in v])))
    q_mag = {k: M.quadrant_stats(v) for k, v in by_mag.items()}
    # 图源×象限 (图依赖 only)
    by_src = collections.defaultdict(list)
    for (ds, sc, m, fam, st, src), v in cells.items():
        if m in GRAPH_DEP:
            by_src[src].append((np.mean([a for a, _ in v]),
                                np.mean([h for _, h in v])))
    q_src = {k: M.quadrant_stats(v) for k, v in by_src.items()}
    out['quadrant'] = dict(all=q_all, by_method=q_method, by_mag=q_mag,
                           by_graph_source=q_src)

    # ---------- 2) 退化曲线 (膝盖): 扰动强度 -> ACR / hit ----------
    # hit of perturbed graphs 用 cell 的 hit3 (= phi(G_base) 的 hit; 扰动后 hit 未存)
    # -> 膝盖分析用 ACR 退化 + E2 补扰动后 hit; 此处先给 ACR 膝盖
    curves = collections.defaultdict(list)
    for (ds, sc, m, fam, st, src), v in cells.items():
        curves[(m, fam, src, sc)].append((st, np.mean([a for a, _ in v])))
    knee = {}
    for k, v in curves.items():
        agg = collections.defaultdict(list)
        for st, a in v:
            agg[st].append(a)
        xs = sorted(agg)
        ys = [np.mean(agg[x]) for x in xs]
        # 膝盖 = 最大二阶差分点 (曲率最大)
        if len(xs) >= 3:
            d2 = np.diff(ys, 2)
            knee_x = xs[int(np.argmin(d2)) + 1]
        else:
            knee_x = None
        knee['|'.join(map(str, k))] = dict(xs=xs, ys=[float(y) for y in ys],
                                           knee=knee_x)
    out['acr_curves'] = knee

    # ---------- 3) hit 主表: 方法×图源×幅度 ----------
    table = collections.defaultdict(list)
    for (ds, sc, m, src), v in hit_cells.items():
        mag = ds.split('_m')[-1]
        table[(m, src, mag, sc)].append(np.mean([x[1] for x in v]))   # hit3
    out['hit_table'] = {'|'.join(map(str, k)): float(np.mean(v))
                        for k, v in sorted(table.items())}

    # ---------- 4) 双参照 ACR (true 学术 vs base 部署) ----------
    # base=true 的 cell: acr3_base 即学术参照; base=pcmci_*: 部署参照
    dual = collections.defaultdict(list)
    for (ds, sc, m, fam, st, src), v in cells.items():
        kind = 'academic(true)' if src == 'true' else f'deploy({src})'
        dual[(m, kind)].append(np.mean([a for a, _ in v]))
    out['acr_dual'] = {'|'.join(map(str, k)): float(np.mean(v))
                       for k, v in sorted(dual.items())}

    # ---------- 5) 图源传导: hit(phi(G)) 随图源变化 (方法×幅度) ----------
    trans = collections.defaultdict(list)
    for (ds, sc, m, src), v in hit_cells.items():
        if m not in GRAPH_DEP:
            continue
        mag = ds.split('_m')[-1]
        trans[(m, mag, src)].append(np.mean([x[1] for x in v]))
    out['graph_transmission'] = {'|'.join(map(str, k)): float(np.mean(v))
                                 for k, v in sorted(trans.items())}

    with open(os.path.join(CACHE, 'a1_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)

    # ---------- 控制台报告 ----------
    print("### 1. Go/No-Go: quadrant (cell-level, ACR>0.8 & hit@3<0.4)")
    print(f"  ALL: stable-wrong={q_all['stable_wrong']:.3f} (n={q_all['n']})")
    for m, q in q_method.items():
        print(f"  {m:14s} stable-wrong={q['stable_wrong']:.3f} "
              f"stable-right={q['stable_right']:.3f} (n={q['n']})")
    print("  by magnitude:", {k: round(v['stable_wrong'], 3) for k, v in q_mag.items()})
    print("  by graph-source:", {k: round(v['stable_wrong'], 3)
                                 for k, v in q_src.items()})
    print("\n### 2. Graph transmission (hit@3 by source, graph-dep methods)")
    for m in GRAPH_DEP:
        for mag in ('0.25', '0.5', '1.5'):
            row = {src: out['graph_transmission'].get(f'{m}|{mag}|{src}')
                   for src in ('true', 'pcmci_clean', 'pcmci_anom')}
            if any(v is not None for v in row.values()):
                print(f"  {m:13s} mag={mag}: " +
                      " ".join(f"{k}={v:.2f}" for k, v in row.items() if v is not None))
    print("\n### 3. ACR curves (knee) sample: PropRank del")
    for k, v in knee.items():
        if k.startswith('PropRank|del|true'):
            print(f"  {k}: xs={v['xs']} ys={[round(y,2) for y in v['ys']]} knee={v['knee']}")
    print("\n-> a1_analysis.json")


if __name__ == '__main__':
    main()
