# -*- coding: utf-8 -*-
"""E88: WADI 图源互换轴 (#4). 干净池学习图 vs 污染学习图(池+攻击段)
vs PCMCI(若时限内可行). 图依赖三族在两种图源下的 hit@3 与 ACR@3,
检验 source bias (传输型方法应受损, 伪稳方法应不敏感).
-> _cache/e88_wadi_swap.json
"""
import os
import sys
import json
import signal

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from evaluation import metrics as M
from scorers import make_scorer
from scripts.e1_anchor import perturb_graph, stable_seed
from scripts.e84_wadi import (load_attack, load_pool_rows, ATTACK_ROOTS,
                              segments_from_label, WIN, N_POOL,
                              CAP_PER_SEG)

CACHE = os.path.join(SRC, '_cache')
CORR_TH = 0.35


def build():
    Xa, lab, sigs = load_attack()
    with np.errstate(invalid='ignore'):
        keep = ~((np.isnan(Xa).mean(0) > 0.99)
                 | (np.nanmax(Xa, 0) == np.nanmin(Xa, 0)))
    sigs = [s for s, k in zip(sigs, keep) if k]
    Xa = Xa[:, keep]
    lab = lab
    Xp = load_pool_rows(sigs)
    with np.errstate(invalid='ignore'):
        keep2 = (np.nanstd(Xp, 0) > 1e-9) & (np.isnan(Xp).mean(0) <= 0.99)
    sigs = [s for s, k in zip(sigs, keep2) if k]
    Xa, Xp = Xa[:, keep2], Xp[:, keep2]
    col_idx = {s: i for i, s in enumerate(sigs)}
    mu, sd = np.nanmean(Xp, 0), np.nanstd(Xp, 0)
    sd[sd == 0] = 1.0
    Xa = np.nan_to_num(np.clip((Xa - mu) / sd, -8, 8)).astype(np.float32)
    Xp = np.nan_to_num(np.clip((Xp - mu) / sd, -8, 8)).astype(np.float32)
    starts = np.linspace(0, len(Xp) - WIN - 1, N_POOL + 2).astype(int)[1:-1]
    X_pool = np.stack([Xp[s:s + WIN] for s in starts]).astype(np.float32)
    segs = segments_from_label(lab)
    X_anom, R_anom = [], []
    for k, (a, b) in enumerate(segs):
        roots = ATTACK_ROOTS[k] if k < len(ATTACK_ROOTS) else None
        if roots is None:
            continue
        ridx = sorted({col_idx[c] for c in roots if c in col_idx})
        if not ridx:
            continue
        n_win = max(0, (b - a - WIN) // 30 + 1)
        take = min(n_win, CAP_PER_SEG)
        step = max(1, n_win // take) if n_win > take else 1
        cnt = 0
        for s in range(a, b - WIN + 1, 30 * step):
            if cnt >= take:
                break
            X_anom.append(Xa[s:s + WIN])
            R_anom.append(set(ridx))
            cnt += 1
    return (np.stack(X_anom).astype(np.float32), R_anom, X_pool,
            Xp, starts, Xa, lab, sigs)


def corr_graph(Xseries):
    C = np.corrcoef(Xseries.T)
    A = (np.abs(C) > CORR_TH).astype(float)
    np.fill_diagonal(A, 0)
    return A


class TimeOut(Exception):
    pass


def _alarm(signum, frame):
    raise TimeOut()


def main():
    X_anom, R_anom, X_pool, Xp, starts, Xa, lab, sigs = build()
    print('windows', len(X_anom), 'd', len(sigs), flush=True)

    graphs = {}
    graphs['corr_pool'] = corr_graph(Xp[starts[0]:starts[-1] + WIN])
    # 污染: 池序列 + 全部攻击段序列
    at_rows = Xa[lab == -1.0]
    mixed = np.vstack([Xp[starts[0]:starts[-1] + WIN], at_rows])
    graphs['corr_contam'] = corr_graph(mixed)
    print('corr_pool edges', int(graphs['corr_pool'].sum()),
          '| corr_contam edges', int(graphs['corr_contam'].sum()),
          flush=True)

    # PCMCI 干净池 + 污染(池+攻击行), 无闹钟直接跑 (Windows 无 SIGALRM)
    try:
        from graphs.sources import pcmci_graph
        A_pc = pcmci_graph(Xp[starts[0]:starts[0] + 80000],
                           tau_max=1)['adj']
        graphs['pcmci_pool'] = np.asarray(A_pc, dtype=float)
        print('pcmci_pool edges', int(graphs['pcmci_pool'].sum()),
              flush=True)
        A_pc2 = pcmci_graph(mixed[-80000:], tau_max=1)['adj']
        graphs['pcmci_contam'] = np.asarray(A_pc2, dtype=float)
        print('pcmci_contam edges', int(graphs['pcmci_contam'].sum()),
              flush=True)
    except Exception as e:
        print('pcmci skipped:', repr(e)[:100], flush=True)

    scorer = make_scorer('iforest').fit(X_pool)
    ctx0 = Context(X_pool, scorer=scorer)
    out = {'graph_edges': {k: int(v.sum()) for k, v in graphs.items()},
           'methods': {}}
    for gname, A in graphs.items():
        for mname, fn in GRAPH_DEPENDENT.items():
            phi0 = fn(X_anom, ctx0.with_graph(A))
            h = M.hit_at_k(phi0, R_anom, 3)
            cons = []
            for mi in range(8):
                rng = np.random.default_rng(
                    stable_seed('e88', gname, mname, mi))
                A_m = perturb_graph(A, ['del', 'add', 'rew'][mi % 3],
                                    0.2, rng)
                cons.append(M.acr_at_k(fn(X_anom, ctx0.with_graph(A_m)),
                                       phi0, 3))
            c = np.stack(cons).mean(0)
            out['methods'].setdefault(mname, {})[gname] = dict(
                hit3=float(np.mean(h)), acr3=float(np.mean(c)),
                sw=float(((c > 0.8) & (h == 0)).mean()))
            print(gname, mname, out['methods'][mname][gname], flush=True)

    path = os.path.join(CACHE, 'e88_wadi_swap.json')
    json.dump(out, open(path, 'w'), indent=1)
    print('->', path)


if __name__ == '__main__':
    main()
