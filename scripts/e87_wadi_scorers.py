# -*- coding: utf-8 -*-
"""E87: WADI 打分器轴演练 (#2+#3): pca 与深度 AE 打分器下的九族网格.

与 e84 同协议(仅换 scorer), 输出每方法 hit@3 / ACR@3 / SW
-> _cache/e87_wadi_scorers.json
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graph_free import GRAPH_FREE
from evaluation import metrics as M
from graphs.graph_ops import graph_features  # noqa: F401
from scorers import make_scorer, TorchPCAScorer
from scripts.e1_anchor import perturb_graph, stable_seed
from scripts.e84_wadi import (load_attack, load_pool_rows, ATTACK_ROOTS,
                              segments_from_label, WIN, N_POOL, CORR_TH,
                              CAP_PER_SEG)

CACHE = os.path.join(SRC, '_cache')


def build():
    Xa, lab, sigs = load_attack()
    nan_mask = np.isnan(Xa).mean(0) > 0.99
    with np.errstate(invalid='ignore'):
        const_mask = np.nanmax(Xa, 0) == np.nanmin(Xa, 0)
    keep = ~(nan_mask | const_mask)
    sigs = [s for s, k in zip(sigs, keep) if k]
    Xa = Xa[:, keep]
    Xp = load_pool_rows(sigs)
    with np.errstate(invalid='ignore'):
        std_p = np.nanstd(Xp, 0)
        nan_p = np.isnan(Xp).mean(0) > 0.99
    keep2 = (std_p > 1e-9) & (~nan_p)
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
    flat = Xp[starts[0]:starts[-1] + WIN]
    C = np.corrcoef(flat.T)
    A = (np.abs(C) > CORR_TH).astype(float)
    np.fill_diagonal(A, 0)
    return np.stack(X_anom).astype(np.float32), R_anom, X_pool, Xp, starts, A


def run_scorer(tag, X_anom, R_anom, X_pool, A, scorer, torch_scorer=None):
    ctx0 = Context(X_pool, scorer=scorer)
    if torch_scorer is not None:
        ctx0.torch_scorer = torch_scorer
    mad = ctx0.pool_stats['mad']
    out = {}
    for mname, fn in list(GRAPH_DEPENDENT.items()):
        phi0 = fn(X_anom, ctx0.with_graph(A))
        cons = []
        for mi in range(8):
            rng = np.random.default_rng(stable_seed('e87', tag, mname, mi))
            A_m = perturb_graph(A, ['del', 'add', 'rew'][mi % 3], 0.2, rng)
            cons.append(M.acr_at_k(fn(X_anom, ctx0.with_graph(A_m)),
                                   phi0, 3))
        c = np.stack(cons).mean(0)
        h = M.hit_at_k(phi0, R_anom, 3)
        out[mname] = dict(hit3=float(np.mean(h)), acr3=float(np.mean(c)),
                          sw=float(((c > 0.8) & (h == 0)).mean()))
        print(tag, mname, out[mname], flush=True)
    for mname, fn in GRAPH_FREE.items():
        phi0 = fn(X_anom, ctx0)
        cons = []
        for mi in range(8):
            rng = np.random.default_rng(stable_seed('e87', tag, mname, mi))
            X_m = (X_anom + rng.normal(0, 1, X_anom.shape).astype(np.float32)
                   * (0.2 * mad)[None, None, :]).astype(np.float32)
            cons.append(M.acr_at_k(fn(X_m, ctx0), phi0, 3))
        c = np.stack(cons).mean(0)
        h = M.hit_at_k(phi0, R_anom, 3)
        out[mname] = dict(hit3=float(np.mean(h)), acr3=float(np.mean(c)),
                          sw=float(((c > 0.8) & (h == 0)).mean()))
        print(tag, mname, out[mname], flush=True)
    return out


def main():
    X_anom, R_anom, X_pool, Xp, starts, A = build()
    print('windows', len(X_anom), flush=True)
    results = {}

    # --- pca scorer ---
    pca = make_scorer('pca').fit(X_pool)
    results['pca'] = run_scorer('pca', X_anom, R_anom, X_pool, A, pca,
                                torch_scorer=TorchPCAScorer(pca))

    # --- deep AE scorer (e8 的 AEScorer) ---
    from scripts.e8_round6 import AEScorer
    ae = AEScorer(seed=0).fit(X_pool)
    results['ae'] = run_scorer('ae', X_anom, R_anom, X_pool, A, ae,
                               torch_scorer=ae)

    # --- (可选) SWaT AE: 若加载器可用 ---
    try:
        from scripts.e14_swat_ci_pred import build_swat_units
        u = build_swat_units()
        from scorers import detection_auroc
        ctx_s = Context(u['X_pool'], scorer=ae.__class__(seed=0).fit(
            u['X_pool']))
        A_s = u['silver_adj']
        res_s = {}
        for mname, fn in GRAPH_DEPENDENT.items():
            phi0 = fn(u['X_anom'], ctx_s.with_graph(A_s))
            h = M.hit_at_k(phi0, u['R_anom'], 3)
            cons = []
            for mi in range(8):
                rng = np.random.default_rng(
                    stable_seed('e87s', mname, mi))
                A_m = perturb_graph(A_s, ['del', 'add', 'rew'][mi % 3],
                                    0.2, rng)
                cons.append(M.acr_at_k(
                    fn(u['X_anom'], ctx_s.with_graph(A_m)), phi0, 3))
            c = np.stack(cons).mean(0)
            res_s[mname] = dict(hit3=float(np.mean(h)),
                                acr3=float(np.mean(c)),
                                sw=float(((c > 0.8) & (h == 0)).mean()))
        results['swat_ae'] = res_s
        print('swat_ae done', flush=True)
    except Exception as e:
        print('swat_ae skipped:', repr(e)[:120], flush=True)

    path = os.path.join(CACHE, 'e87_wadi_scorers.json')
    json.dump(results, open(path, 'w'), indent=1)
    print('->', path)


if __name__ == '__main__':
    main()
