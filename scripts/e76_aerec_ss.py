# -*- coding: utf-8 -*-
"""E76: AERec 上 SS (元宝二审关键空缺) + 关系族配对检验.

(a) AERec (图无关 AE 重构贡献) 在 SS 全部 72 单元, 协议对齐 e66
    (iforest 打分器, 无图轴), 输出并入故障族分层
(b) DPTA-G vs zDev 关系族 (delay/loss) 配对置换检验 + CI
输出: _cache/e76_aerec_ss.json
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
import sys
import json
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.rcaeval2 import load_units
from scorers import make_scorer
from attribution import Context
from attribution.graph_free import GRAPH_FREE
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')


def main():
    t0 = time.time()
    units = [u for u in load_units('ss', reps=(1, 2, 3))
             if not np.isnan(u['series_full']).any()
             and not np.isnan(u['series_normal']).any()]
    print(f'{len(units)} units', flush=True)
    recs = []
    for ui, u in enumerate(units):
        scorer = make_scorer('iforest').fit(u['X_pool'])
        ctx0 = Context(u['X_pool'], scorer=scorer)
        phi = GRAPH_FREE['AERec'](u['X_anom'], ctx0)
        hit3 = float(M.mean_hit(phi, u['R_anom'], 3))
        recs.append(dict(unit=u['name'], fault=u['fault'], hit3=hit3))
        if (ui + 1) % 12 == 0 or ui == len(units) - 1:
            print(f'[{ui+1}/{len(units)}] {u["name"]} hit3={hit3:.3f} '
                  f'({time.time()-t0:.0f}s)', flush=True)
    json.dump(recs, open(os.path.join(CACHE, 'e76_aerec_ss.json'), 'w'),
              default=float, indent=1)

    # 分层汇总 + 配对检验
    e66 = json.load(open(os.path.join(CACHE, 'e66_rcaeval2.json')))
    dpt = {}
    zd = {}
    for r in e66:
        if r.get('method') == 'DPTA-G' and r['graph_source'] == 'pcmci_clean':
            dpt[r['unit']] = r['hit3']
        if r.get('method') == 'zDev':
            zd[r['unit']] = r['hit3']
    fam = lambda n: ('deviational' if n.split('_')[-2] in ('cpu', 'mem')
                     else 'relational' if n.split('_')[-2] in ('delay', 'loss')
                     else 'disk')
    import collections
    agg = collections.defaultdict(list)
    for r in recs:
        agg[fam(r['unit'])].append(r['hit3'])
    print('\n=== AERec on SS (hit@3 by fault family) ===')
    for f in ('deviational', 'relational', 'disk'):
        print(f'  {f:12} {np.mean(agg[f]):.3f} (n={len(agg[f])})')
    # AERec vs zDev vs DPTA-G 关系族配对
    rel = [r for r in recs if fam(r['unit']) == 'relational']
    a = [r['hit3'] for r in rel]
    b = [zd[r['unit']] for r in rel]
    c = [dpt[r['unit']] for r in rel]
    rng = np.random.default_rng(20260909)
    def pp(x, y, tag):
        d = np.array(x) - np.array(y)
        obs = d.mean()
        null = (rng.choice((-1, 1), (10000, len(d))) * d).mean(1)
        p = (np.abs(null) >= abs(obs)).mean()
        print(f'  {tag}: diff={obs:+.3f} p={p:.4f}')
    print('\n=== relational 族配对置换 (n=%d) ===' % len(rel))
    pp(a, b, 'AERec-zDev ')
    pp(c, b, 'DPTA-G-zDev')
    pp(a, c, 'AERec-DPTAG')


if __name__ == '__main__':
    main()
