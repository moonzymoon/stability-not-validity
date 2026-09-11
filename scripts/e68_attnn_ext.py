# -*- coding: utf-8 -*-
"""E68 (夜航 P2): AttNN 跨测试床扩展 — SS 72 单元 + OB-holdout 17 单元.

e65 只在 SCM 上评了 AttNN; 本脚本把同一协议 (被解释窗口拟合的注意力代理,
phi=α, 输入扰动 ACR@3, M=8, noise 0.3/slice 2) 扩到两个微服务测试床,
使九族网格的深度基线拥有三个测试床观测点。种子=MASTER_SEED (20260906),
逐单元确定性。输出: _cache/e68_attnn_ext.json
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
from data.rcaeval import load_ob_units
from scorers import make_scorer
from attribution import Context
from attribution.attnn import attnn_alpha, MASTER_SEED
from scripts.e1_anchor import stable_seed, perturb_input
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')
M_PERT = 8


def run_unit(u, tag, sk):
    t0 = time.time()
    scorer = make_scorer(sk).fit(u['X_pool'])
    ctx0 = Context(u['X_pool'], scorer=scorer)
    alpha_fn, info = attnn_alpha(u['X_anom'], ctx0, seed=MASTER_SEED)
    phi = alpha_fn(u['X_anom'])
    hit3 = float(M.mean_hit(phi, u['R_anom'], 3))
    acrs = []
    for fam, sts in (dict(noise=[0.3], slice=[2])).items():
        for st in sts:
            for mi in range(M_PERT):
                rng = np.random.default_rng(stable_seed(
                    'e68', tag, u['name'], sk, fam, st, mi))
                Xm = perturb_input(u['X_anom'], fam, st, rng, ctx0)
                pm = alpha_fn(Xm)
                acrs.append(float(M.acr_at_k(pm, phi, 3).mean()))
    return dict(testbed=tag, unit=u['name'], scorer=sk, method='AttNN',
                hit3=hit3, acr3_input=float(np.mean(acrs)),
                fidelity=info['val_corr'],
                wall_s=round(time.time() - t0, 1))


def main():
    t_all = time.time()
    records = []
    ss_units = [u for u in load_units('ss', reps=(1, 2, 3))
                if not np.isnan(u['series_full']).any()
                and not np.isnan(u['series_normal']).any()]
    obh_units = [u for u in load_ob_units(reps=(4, 5))
                 if u['fault'] in ('cpu', 'mem')
                 and not np.isnan(u['series_full']).any()]
    print(f'units: ss={len(ss_units)} obh={len(obh_units)}', flush=True)
    for tag, units in (('ss', ss_units), ('ob_holdout', obh_units)):
        for ui, u in enumerate(units):
            for sk in ('iforest', 'pca'):
                r = run_unit(u, tag, sk)
                records.append(r)
            if (ui + 1) % 10 == 0 or ui == len(units) - 1:
                json.dump(records, open(os.path.join(
                    CACHE, 'e68_attnn_ext.json.part'), 'w'), default=float)
                print(f'[{tag} {ui+1}/{len(units)}] last: '
                      f'hit3={r["hit3"]:.2f} acr={r["acr3_input"]:.2f} '
                      f'({records[-1]["wall_s"]}s)', flush=True)
    out = os.path.join(CACHE, 'e68_attnn_ext.json')
    os.replace(os.path.join(CACHE, 'e68_attnn_ext.json.part'), out)

    import collections
    agg = collections.defaultdict(list)
    for r in records:
        agg[(r['testbed'], r['scorer'])].append((r['hit3'], r['acr3_input'],
                                                 r['fidelity']))
    print('\n=== E68 AttNN cross-testbed ===')
    for k in sorted(agg, key=str):
        v = agg[k]
        print(f'  {str(k):26} hit3={np.mean([x[0] for x in v]):.3f} '
              f'acr3={np.mean([x[1] for x in v]):.3f} '
              f'fid={np.mean([x[2] for x in v]):.3f} (n={len(v)})')
    print(f'total {time.time()-t_all:.0f}s -> {out}')


if __name__ == '__main__':
    main()
