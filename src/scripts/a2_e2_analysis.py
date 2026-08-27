# -*- coding: utf-8 -*-
"""A2: E2 全量分析 —— 打分器质量传导 / TE 部署场景 / 扰动后 hit 退化(膝盖)。
输出: _cache/a2_analysis.json + 控制台摘要
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


def load(fn):
    p = os.path.join(CACHE, fn)
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def analyze(fn, tag):
    recs = load(fn)
    if recs is None:
        print(f"[skip] {fn}")
        return {}
    out = {}

    # 1) 打分器质量分层: (method, graph_source, frac) -> (det_auroc, hit3) 配对
    qual = collections.defaultdict(list)
    for r in recs:
        if r.get('method', '').startswith('_') or r['family'] != 'none':
            continue
        if 'det_auroc' not in r:
            continue
        qual[(r['method'], r['graph_source'])].append(
            (r['train_frac'], r['det_auroc'], r['hit3']))
    curves = {}
    for k, v in sorted(qual.items()):
        byf = collections.defaultdict(list)
        for f, au, h in v:
            byf[f].append((au, h))
        xs, au_m, h_m = [], [], []
        for f in sorted(byf):
            xs.append(f)
            au_m.append(float(np.mean([a for a, _ in byf[f]])))
            h_m.append(float(np.mean([h for _, h in byf[f]])))
        curves['|'.join(map(str, k))] = dict(frac=xs, auroc=au_m, hit=h_m)
    out['quality_curves'] = curves

    # 质量敏感性: 每 (method) 的 det_auroc 变化 vs hit 变化 (frac 0.25 -> 1.0)
    sens = {}
    for k, c in curves.items():
        if len(c['frac']) >= 2 and c['frac'][0] == 0.25:
            dau = c['auroc'][-1] - c['auroc'][0]
            dhit = c['hit'][-1] - c['hit'][0]
            sens[k] = dict(delta_auroc=float(dau), delta_hit=float(dhit))
    out['quality_sensitivity'] = sens

    # 2) TE 部署场景 (无真值图): 方法 ACR_base + hit
    if tag == 'te':
        te = collections.defaultdict(lambda: dict(h=[], a=[], hp=[]))
        for r in recs:
            if r.get('method', '').startswith('_'):
                continue
            k = (r['method'], r['scorer'])
            if r['family'] == 'none':
                te[k]['h'].append(r['hit3'])
            elif r.get('acr3_base') is not None:
                te[k]['a'].append(r['acr3_base'])
                if r.get('hit3_pert') is not None:
                    te[k]['hp'].append(r['hit3_pert'])
        out['te_deployment'] = {
            f'{m}|{s}': dict(hit3=float(np.mean(v['h'])) if v['h'] else None,
                             acr3=float(np.mean(v['a'])) if v['a'] else None,
                             hit3_pert=float(np.mean(v['hp'])) if v['hp'] else None)
            for (m, s), v in te.items()}

    # 3) 扰动后 hit 退化 (膝盖, 随机扰动族): (method, fam, source) -> hit vs strength
    deg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in recs:
        if r.get('method', '').startswith('_') or r['family'] == 'none':
            continue
        if r.get('hit3_pert') is None:
            continue
        deg[(r['method'], r['family'], r['graph_source'])][r['strength']].append(
            r['hit3_pert'])
    knee = {}
    for k, bys in deg.items():
        xs = sorted(bys)
        ys = [float(np.mean(bys[x])) for x in xs]
        kn = None
        if len(xs) >= 3:
            d2 = np.diff(ys, 2)
            kn = xs[int(np.argmin(d2)) + 1]
        knee['|'.join(map(str, k))] = dict(xs=xs, ys=ys, knee=kn)
    out['hit_degradation'] = knee

    # 4) 象限 (含 hit3_pert 版本: 稳定=ACR>0.8 且 扰动后hit<0.4)
    cells = collections.defaultdict(list)
    for r in recs:
        if r.get('method', '').startswith('_') or r['family'] == 'none':
            continue
        if r.get('acr3_base') is None:
            continue
        cells[(r['dataset'], r['scorer'], r['method'], r['family'],
               r['strength'], r['graph_source'])].append(
            (r['acr3_base'], r['hit3']))
    pts = [(np.mean([a for a, _ in v]), np.mean([h for _, h in v]))
           for v in cells.values()]
    out['quadrant'] = M.quadrant_stats(pts)
    return out


def main():
    out = dict(scm=analyze('e2_scm_quality.json', 'scm'),
               te=analyze('e2_te.json', 'te'))
    with open(os.path.join(CACHE, 'a2_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)

    print("### quality sensitivity (frac 0.25->1.0): delta AUC vs delta hit@3")
    for k, v in list(out['scm'].get('quality_sensitivity', {}).items())[:12]:
        print(f"  {k}: dAUC={v['delta_auroc']:+.3f} dhit={v['delta_hit']:+.3f}")
    if out['te']:
        print("\n### TE deployment (hit@3 / ACR@3 / hit@3 after perturb)")
        for k, v in sorted(out['te'].get('te_deployment', {}).items()):
            print(f"  {k}: hit={v['hit3']} acr={v['acr3']} hit_pert={v['hit3_pert']}")
    print("\n### SCM quadrant (E2 with scorer stratification)")
    print(out['scm'].get('quadrant'))
    print("\n### hit degradation knees")
    for k, v in list(out['scm'].get('hit_degradation', {}).items())[:9]:
        print(f"  {k}: ys={[round(y,2) for y in v['ys']]} knee={v['knee']}")
    print("-> a2_analysis.json")


if __name__ == '__main__':
    main()
