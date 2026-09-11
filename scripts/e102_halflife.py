# -*- coding: utf-8 -*-
"""E102: 可靠性半衰期 (措施三) --- WADI 池龄效应.

正常池固定(正常文件, t0), 攻击段按时间排序(14段跨~45h). 每段独立计算
hit@3 / ACR@3 (4方法), 看效度是否随池龄衰减而稳定性持平.
-> _cache/e102_halflife.json
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
from scripts.e84_wadi import (load_attack, load_pool_rows, ATTACK_ROOTS,
                              segments_from_label, WIN, N_POOL, CAP_PER_SEG)

CACHE = os.path.join(SRC, '_cache')
CORR_TH = 0.35


def build():
    Xa, lab, sigs = load_attack()
    with np.errstate(invalid='ignore'):
        keep = ~((np.isnan(Xa).mean(0) > 0.99)
                 | (np.nanmax(Xa, 0) == np.nanmin(Xa, 0)))
    sigs = [s for s, k in zip(sigs, keep) if k]
    Xa = Xa[:, keep]
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
    per_seg = []
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
        Xw, Rw = [], []
        cnt = 0
        for s in range(a, b - WIN + 1, 30 * step):
            if cnt >= take:
                break
            Xw.append(Xa[s:s + WIN])
            Rw.append(set(ridx))
            cnt += 1
        if Xw:
            per_seg.append(dict(seg=k, t_start=a, t_end=b,
                                X=np.stack(Xw).astype(np.float32), R=Rw))
    flat = Xp[starts[0]:starts[-1] + WIN]
    C = np.corrcoef(flat.T)
    A = (np.abs(C) > CORR_TH).astype(float)
    np.fill_diagonal(A, 0)
    return per_seg, X_pool, A


def main():
    per_seg, X_pool, A = build()
    scorer = make_scorer('iforest').fit(X_pool)
    ctx0 = Context(X_pool, scorer=scorer)
    mad = ctx0.pool_stats['mad']
    out = {'segments': []}
    methods = (('PropRank', GRAPH_DEPENDENT['PropRank'], True),
               ('DPTA-G', GRAPH_DEPENDENT['DPTA-G'], True),
               ('zDev', GRAPH_FREE['zDev'], False),
               ('AERec', GRAPH_FREE['AERec'], False))
    for seg in per_seg:
        X, R = seg['X'], seg['R']
        rec = dict(seg=seg['seg'], t_start=seg['t_start'],
                   n=len(X))
        for mname, fn, graph in methods:
            phi0 = fn(X, ctx0.with_graph(A)) if graph else fn(X, ctx0)
            h = M.hit_at_k(phi0, R, 3)
            cons = []
            for mi in range(8):
                rng = np.random.default_rng(
                    stable_seed('e102', seg['seg'], mname, mi))
                if graph:
                    A_m = perturb_graph(A, ['del', 'add', 'rew'][mi % 3],
                                        0.2, rng)
                    phi_m = fn(X, ctx0.with_graph(A_m))
                else:
                    X_m = (X + rng.normal(0, 1, X.shape).astype(np.float32)
                           * (0.2 * mad)[None, None, :]).astype(np.float32)
                    phi_m = fn(X_m, ctx0)
                cons.append(M.acr_at_k(phi_m, phi0, 3))
            rec[mname] = dict(hit3=float(np.mean(h)),
                              acr3=float(np.mean(cons)))
        out['segments'].append(rec)
        print(f"seg {seg['seg']:2d} t={seg['t_start']:6d} "
              f"zDev h/a={rec['zDev']['hit3']:.2f}/{rec['zDev']['acr3']:.2f} "
              f"Prop h/a={rec['PropRank']['hit3']:.2f}/"
              f"{rec['PropRank']['acr3']:.2f}", flush=True)
    # 池龄相关
    ts = np.array([r['t_start'] for r in out['segments']], dtype=float)
    ts = (ts - ts.min()) / (ts.max() - ts.min() + 1e-9)
    out['age_correlation'] = {}
    for mname, _, _ in methods:
        hs = np.array([r[mname]['hit3'] for r in out['segments']])
        as_ = np.array([r[mname]['acr3'] for r in out['segments']])
        out['age_correlation'][mname] = dict(
            hit_r=float(np.corrcoef(ts, hs)[0, 1]),
            acr_r=float(np.corrcoef(ts, as_)[0, 1]))
        print(f"{mname}: corr(age, hit)={out['age_correlation'][mname]['hit_r']:+.2f} "
              f"corr(age, ACR)={out['age_correlation'][mname]['acr_r']:+.2f}",
              flush=True)
    json.dump(out, open(os.path.join(CACHE, 'e102_halflife.json'), 'w'),
              indent=1)
    print('-> e102_halflife.json')


if __name__ == '__main__':
    main()
