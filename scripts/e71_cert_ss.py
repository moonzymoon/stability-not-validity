# -*- coding: utf-8 -*-
"""E71 (夜航 P5): 经验紧包络证书扩展到 SS 第四测试床 (port 自 e62).

协议与 e62 (TE) 完全一致: base = pcmci_clean 学图 (部署场景, 图缓存复用 e66),
PropRank/DPTA-G, M=8 扰动 (q=0.1 循环 del/add/rew), 4+4 估计/评估分割,
  cert_worst: gap > 2*max(est半);  cert_emp50: gap > 2*median(est半)
评估半真实 top-3 翻转率. SS 全部 72 无 NaN 单元.
输出: _cache/e71_cert_ss.json
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
from attribution.graph_dep import GRAPH_DEPENDENT
from evaluation import metrics as M
from scripts.e1_anchor import perturb_graph, stable_seed

CACHE = os.path.join(SRC, '_cache')
Q = 0.1
M_INST = 8
N_EST = 4


def run():
    t0 = time.time()
    units = [u for u in load_units('ss', reps=(1, 2, 3))
             if not np.isnan(u['series_full']).any()
             and not np.isnan(u['series_normal']).any()]
    print(f'{len(units)} units', flush=True)
    rows = []
    for ui, u in enumerate(units):
        gpath = os.path.join(CACHE, f'e66_graph_clean_{u["name"]}.npz')
        A_clean = np.load(gpath)['A'].astype(float)
        scorer = make_scorer('iforest').fit(u['X_pool'])
        ctx0 = Context(u['X_pool'], scorer=scorer)
        ctx_g = ctx0.with_graph(A_clean)
        for mname in ('PropRank', 'DPTA-G'):
            fn = GRAPH_DEPENDENT[mname]
            phi0 = fn(u['X_anom'], ctx_g)
            hits = M.hit_at_k(phi0, u['R_anom'], 3)
            phis = []
            for mi in range(M_INST):
                rng = np.random.default_rng(stable_seed('e71', u['name'],
                                                        mname, mi))
                A_m = perturb_graph(A_clean, ['del', 'add', 'rew'][mi % 3],
                                    Q, rng)
                phis.append(fn(u['X_anom'], ctx0.with_graph(A_m)))
            phis = np.stack(phis)
            D = np.abs(phis - phi0[None])
            eta = D.max(axis=2).T
            order = np.argsort(-phi0, axis=1)
            gap = phi0[np.arange(len(phi0)), order[:, 2]] - \
                phi0[np.arange(len(phi0)), order[:, 3]]
            sets0 = [set(np.argsort(-r)[:3]) for r in phi0]
            flip = np.zeros(len(phi0))
            for mi in range(N_EST, M_INST):
                sm = [set(np.argsort(-r)[:3]) for r in phis[mi]]
                flip += np.array([a != b for a, b in zip(sm, sets0)])
            flip /= (M_INST - N_EST)
            est = eta[:, :N_EST]
            for i in range(len(phi0)):
                rows.append(dict(
                    unit=u['name'], method=mname, window=i,
                    label=int(hits[i]), gap=float(gap[i]),
                    eta_max_est=float(est[i].max()),
                    flip_eval=float(flip[i]),
                    cert_worst=bool(gap[i] > 2 * est[i].max()),
                    cert_emp50=bool(gap[i] > 2 * np.median(est[i]))))
        if (ui + 1) % 12 == 0 or ui == len(units) - 1:
            json.dump(rows, open(os.path.join(
                CACHE, 'e71_cert_ss.json.part'), 'w'), default=float)
            print(f'[{ui+1}/{len(units)}] {u["name"]} ({time.time()-t0:.0f}s)',
                  flush=True)
    out = os.path.join(CACHE, 'e71_cert_ss.json')
    os.replace(os.path.join(CACHE, 'e71_cert_ss.json.part'), out)

    print('\n=== SS 经验证书 (pcmci_clean, q=0.1, 4+4) ===')
    for m in ('PropRank', 'DPTA-G'):
        sub = [r for r in rows if r['method'] == m]
        base_flip = np.mean([r['flip_eval'] for r in sub])
        print(f'{m}: n={len(sub)} 基线评估半翻转率 {base_flip:.3f}')
        for c in ('cert_worst', 'cert_emp50'):
            cov = np.mean([r[c] for r in sub])
            fl = [r['flip_eval'] for r in sub if r[c]]
            fls = f'{np.mean(fl):.3f}' if fl else 'NA'
            wrong = [r['label'] for r in sub if r[c]]
            ws = f'{np.mean(wrong):.3f}' if wrong else 'NA'
            print(f'  {c}: 覆盖 {cov*100:.1f}%, 证内翻转率 {fls}, '
                  f'证内错误率(1-hit) {ws}')


if __name__ == '__main__':
    run()
