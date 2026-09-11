# -*- coding: utf-8 -*-
"""E62: 完备2 — 经验紧包络证书扩展到 TE 第三测试床.
TE 6 单元 (IDV1/4/11 × 2), base = pcmci_clean 学图 (部署场景),
PropRank/DPTA-G, M=8 扰动 (q=0.1 循环 del/add/rew), 4+4 估计/评估分割:
  worst-case: gap > 2*max(est半);  empirical: gap > 2*median(est半)
  评估半真实 top-3 翻转率. 与 SCM 的 33%→70% 结果对照.
输出: _cache/e62_te_cert.json
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.tep import load_tep_units
from graphs.sources import pcmci_graph
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from evaluation import metrics as M
from scripts.e1_anchor import perturb_graph, stable_seed

CACHE = os.path.join(SRC, '_cache')
Q = 0.1
M_INST = 8
N_EST = 4


def graph_cached(series, cache_name, tau_max=2):
    path = os.path.join(CACHE, cache_name)
    if os.path.exists(path):
        d = np.load(path)
        return d['A'], d['val']
    r = pcmci_graph(series, tau_max=tau_max)
    val = np.abs(r['val_matrix']).max(2)
    val = val / (val.max() + 1e-9)
    np.savez(path, A=r['adj'], val=val)
    return r['adj'], val


def run():
    rows = []
    for u in load_tep_units():
        name = u['name']
        A_clean, _ = graph_cached(u['series_normal'],
                                  f'e24_graph_clean_{name}.npz')
        scorer = make_scorer('iforest').fit(u['X_pool'])
        ctx0 = Context(u['X_pool'], scorer=scorer)
        ctx_g = ctx0.with_graph(A_clean)
        for mname in ('PropRank', 'DPTA-G'):
            fn = GRAPH_DEPENDENT[mname]
            phi0 = fn(u['X_anom'], ctx_g)
            hits = M.hit_at_k(phi0, u['R_anom'], 3)
            phis = []
            for mi in range(M_INST):
                rng = np.random.default_rng(stable_seed('e62', name,
                                                        mname, mi))
                A_m = perturb_graph(A_clean, ['del', 'add', 'rew'][mi % 3],
                                    Q, rng)
                phis.append(fn(u['X_anom'], ctx0.with_graph(A_m)))
            phis = np.stack(phis)
            D = np.abs(phis - phi0[None])
            eta = D.max(axis=2).T                       # (n, M)
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
                    unit=name, method=mname, window=i,
                    label=int(hits[i]), gap=float(gap[i]),
                    eta_max_est=float(est[i].max()),
                    flip_eval=float(flip[i]),
                    cert_worst=bool(gap[i] > 2 * est[i].max()),
                    cert_emp50=bool(gap[i] > 2 * np.median(est[i]))))
        print(f'{name} done', flush=True)

    json.dump(rows, open(os.path.join(CACHE, 'e62_te_cert.json'), 'w'),
              default=float)
    print('\n=== TE 经验证书 (pcmci_clean, q=0.1, 4+4) ===')
    for m in ('PropRank', 'DPTA-G'):
        sub = [r for r in rows if r['method'] == m]
        base_flip = np.mean([r['flip_eval'] for r in sub])
        print(f'{m}: n={len(sub)} 基线评估半翻转率 {base_flip:.3f}')
        for c in ('cert_worst', 'cert_emp50'):
            cov = np.mean([r[c] for r in sub])
            fl = [r['flip_eval'] for r in sub if r[c]]
            fls = f'{np.mean(fl):.3f}' if fl else 'NA'
            print(f'  {c}: 覆盖 {cov*100:.1f}%, 证内翻转率 {fls}')


if __name__ == '__main__':
    run()
