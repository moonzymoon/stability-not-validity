# -*- coding: utf-8 -*-
"""E99: SMD 标准套件复现 (interpretation_label 作弱逐维真值).

U8 关闭: 标准套件覆盖 2/6 -> 3/7. 协议: 8台分层(1/2/3组), 窗口/步长100,
正常池=训练段, 异常窗=测试段标签区间内整窗, 根因=窗口内解释标记>=5个时间步
的维度(弱真值, 事后通道标记而非注入根因 --- 论文§gt已界定).
-> _cache/e99_smd.json
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from scorers import make_scorer, detection_auroc
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graph_free import GRAPH_FREE
from evaluation import metrics as M
from scripts.e1_anchor import perturb_graph, stable_seed

CACHE = os.path.join(SRC, '_cache')
SMD = os.path.join(SRC, '_external', 'SMD')
MACHINES = ['machine-1-1', 'machine-1-4', 'machine-1-7',
            'machine-2-2', 'machine-2-5', 'machine-2-8',
            'machine-3-1', 'machine-3-6']
WIN = 100
STRIDE = 100
POOL_CAP = 200
SEG_CAP = 40
MARK_TH = 5
CORR_TH = 0.35


def load(m, kind):
    p = os.path.join(SMD, f'{m}_{kind}.txt')
    if kind == 'interpretation_label':
        # 稀疏格式: 每行 "start-end:dim" 或 "t:dim" (dim 1-based)
        n_test = sum(1 for _ in open(os.path.join(
            SMD, f'{m}_test.txt'))) - 1
        d_test = len(open(os.path.join(
            SMD, f'{m}_test.txt')).readline().strip().split(','))
        il = np.zeros((n_test, d_test), dtype=float)
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            rng, dims = line.rsplit(':', 1)
            if '-' in rng:
                a, b = rng.split('-')
                a, b = int(a), int(b)
            else:
                a = b = int(rng)
            for dim in dims.split(','):
                dim = int(dim.strip()) - 1
                if 0 <= dim < d_test:
                    il[a:b + 1, dim] = 1.0
        return il
    return np.loadtxt(p, delimiter=',')


def run_machine(m):
    tr = load(m, 'train').astype(np.float32)
    te = load(m, 'test').astype(np.float32)
    tl = load(m, 'test_label').astype(float)
    il = load(m, 'interpretation_label').astype(float)
    mu, sd = tr.mean(0), tr.std(0)
    keep = sd > 1e-9
    tr, te, il = tr[:, keep], te[:, keep], il[:, keep]
    sd = sd[keep]
    sd[sd == 0] = 1
    tr = (tr - mu[keep]) / sd
    te = (te - mu[keep]) / sd
    te = np.clip(te, -8, 8)
    tr = np.clip(tr, -8, 8)
    d = tr.shape[1]

    rng = np.random.default_rng(abs(hash(m)) % 2**31)
    starts = rng.choice(len(tr) - WIN - 1, POOL_CAP, replace=False)
    X_pool = np.stack([tr[s:s + WIN] for s in starts]).astype(np.float32)

    # 重叠协议: 全测试序列滑动, 保留异常步占比>=30% 且有标记维度的窗口
    lab = tl.squeeze() if tl.ndim > 1 else tl
    Xa, Ra = [], []
    w0 = 0
    while w0 + WIN <= len(lab):
        frac = lab[w0:w0 + WIN].mean()
        if frac >= 0.3:
            marks = il[w0:w0 + WIN].sum(0)
            roots = set(np.where(marks >= MARK_TH)[0].tolist())
            if roots:
                Xa.append(te[w0:w0 + WIN])
                Ra.append(roots)
        w0 += STRIDE
    if not Xa:
        return None
    return np.stack(Xa).astype(np.float32), Ra, X_pool, tr, d


def main():
    acc = {}
    meta = []
    for m in MACHINES:
        r = run_machine(m)
        if r is None:
            print(m, 'skipped (no labeled windows)', flush=True)
            continue
        X_anom, R_anom, X_pool, tr_series, d = r
        scorer = make_scorer('iforest').fit(X_pool)
        auroc = detection_auroc(scorer, X_pool, X_anom)
        ctx0 = Context(X_pool, scorer=scorer)
        C = np.corrcoef(tr_series[:15000].T)
        A = (np.abs(C) > CORR_TH).astype(float)
        np.fill_diagonal(A, 0)
        mad = ctx0.pool_stats['mad']
        meta.append(dict(machine=m, d=d, n_win=len(X_anom),
                         roots_mean=float(np.mean([len(x) for x in R_anom])),
                         det_auroc=float(auroc),
                         graph_edges=int(A.sum())))
        for mname, fn in (('PropRank', GRAPH_DEPENDENT['PropRank']),
                          ('DPTA-G', GRAPH_DEPENDENT['DPTA-G']),
                          ('zDev', GRAPH_FREE['zDev']),
                          ('AERec', GRAPH_FREE['AERec']),
                          ('GlobalCF', GRAPH_FREE['GlobalCF']),
                          ('Random', GRAPH_FREE['Random'])):
            phi0 = fn(X_anom, ctx0.with_graph(A)) \
                if mname in GRAPH_DEPENDENT else fn(X_anom, ctx0)
            cons = []
            for mi in range(8):
                rng = np.random.default_rng(stable_seed('e99', m, mname, mi))
                if mname in GRAPH_DEPENDENT:
                    A_m = perturb_graph(A, ['del', 'add', 'rew'][mi % 3],
                                        0.2, rng)
                    phi_m = fn(X_anom, ctx0.with_graph(A_m))
                else:
                    X_m = (X_anom + rng.normal(0, 1, X_anom.shape)
                           .astype(np.float32)
                           * (0.2 * mad)[None, None, :]).astype(np.float32)
                    phi_m = fn(X_m, ctx0)
                cons.append(M.acr_at_k(phi_m, phi0, 3))
            c = np.stack(cons).mean(0)
            h = M.hit_at_k(phi0, R_anom, 3)
            e = acc.setdefault(mname, [])
            e.append((float(np.mean(h)), float(np.mean(c)),
                      float(((c > 0.8) & (h == 0)).mean()), len(h)))
        print(m, 'done: win', len(X_anom), 'd', d, 'det', round(auroc, 2),
              flush=True)

    out = dict(meta=meta, methods={})
    tot = sum(x['n_win'] for x in meta)
    for mname, es in acc.items():
        w = np.array([e[3] for e in es])
        out['methods'][mname] = dict(
            hit3=float(np.average([e[0] for e in es], weights=w)),
            acr3=float(np.average([e[1] for e in es], weights=w)),
            sw=float(np.average([e[2] for e in es], weights=w)))
        print(f'{mname:10s} {out["methods"][mname]}', flush=True)
    out['total_windows'] = int(tot)
    json.dump(out, open(os.path.join(CACHE, 'e99_smd.json'), 'w'), indent=1)
    print('-> e99_smd.json | total windows', tot)


if __name__ == '__main__':
    main()
