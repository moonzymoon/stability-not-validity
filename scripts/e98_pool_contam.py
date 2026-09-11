# -*- coding: utf-8 -*-
"""E98: 打分器训练侧污染 (U9, 与 ESWA'24/RoCA 同设定).

问题: 论文的污染设定在图源侧(pcmci_anom); 训练侧污染(池中混入异常窗)
对*归因*的后果未量化 --- 同级刊已有多篇训练侧污染方法论文.
设计: SCM medium 5 seeds; pool_contam = 干净池 + 50% 异常窗(模拟"无干净
正常期"); 检测AUROC + 四方法 hit@3 (干净池 vs 污染池训练).
-> _cache/e98_pool_contam.json
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.scm import make_sample, build_windows
from scorers import make_scorer, detection_auroc
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graph_free import GRAPH_FREE
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')


def main():
    out = {'per_seed': [], 'methods': {}}
    acc = {}
    for seed in range(5):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                        seed=seed, noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        X_anom, X_pool, roots = w['X_anom'], w['X_pool'], w['R_anom']
        A = s['adjacency'].astype(float)

        # 污染池: 干净池 + 50% 异常窗
        rng = np.random.default_rng(seed + 9)
        k = len(X_anom) // 2
        contam_idx = rng.choice(len(X_anom), k, replace=False)
        X_pool_c = np.concatenate([X_pool, X_anom[contam_idx]])

        sc_clean = make_scorer('iforest').fit(X_pool)
        sc_contam = make_scorer('iforest').fit(X_pool_c)
        au_clean = detection_auroc(sc_clean, X_pool, X_anom)
        au_contam = detection_auroc(sc_contam, X_pool, X_anom)
        out['per_seed'].append(dict(seed=seed, det_clean=float(au_clean),
                                    det_contam=float(au_contam)))

        for tag, pool, scorer in (('clean', X_pool, sc_clean),
                                  ('contam', X_pool_c, sc_contam)):
            ctx = Context(pool, scorer=scorer)
            for mname, fn in (('DPTA-G', GRAPH_DEPENDENT['DPTA-G']),
                              ('PropRank', GRAPH_DEPENDENT['PropRank']),
                              ('zDev', GRAPH_FREE['zDev']),
                              ('AERec', GRAPH_FREE['AERec'])):
                # AERec 在污染池上自训练(其内部用池), 其余方法只吃 scorer
                phi = fn(X_anom, ctx.with_graph(A)) \
                    if mname in GRAPH_DEPENDENT else fn(X_anom, ctx)
                h = float(M.mean_hit(phi, roots, 3))
                acc.setdefault(mname, {})[tag] = acc.setdefault(
                    mname, {}).get(tag, [])
                acc[mname][tag].append(h)
        print(f'seed {seed}: det {au_clean:.2f}->{au_contam:.2f}',
              flush=True)

    for m, d in acc.items():
        out['methods'][m] = dict(
            hit3_clean=float(np.mean(d['clean'])),
            hit3_contam=float(np.mean(d['contam'])))
        print(f'{m:10s} hit@3 {out["methods"][m]["hit3_clean"]:.2f} -> '
              f'{out["methods"][m]["hit3_contam"]:.2f}', flush=True)
    out['det_mean'] = dict(
        clean=float(np.mean([r['det_clean'] for r in out['per_seed']])),
        contam=float(np.mean([r['det_contam'] for r in out['per_seed']])))
    print('detection AUROC mean: %.2f -> %.2f' % (
        out['det_mean']['clean'], out['det_mean']['contam']), flush=True)
    json.dump(out, open(os.path.join(CACHE, 'e98_pool_contam.json'), 'w'),
              indent=1)
    print('-> e98_pool_contam.json')


if __name__ == '__main__':
    main()
