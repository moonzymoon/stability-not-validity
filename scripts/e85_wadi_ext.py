# -*- coding: utf-8 -*-
"""E85: WADI 三件扩展 (接 e84 管线).

(a) 预测器零样本 SCM->WADI: e3 特征配方(g/c/a/s/d) 逐窗口,
    HGB 训练于 e3_windows(SCM), 零样本评估 WADI 行;
(b) 证书复现: e39 协议 (gamma=top3-top4 gap, eta=max displacement
    over M draws, certified = gamma > 2 eta) 于 DPTA-G/PropRank;
(c) 逐窗口 bootstrap 95% CI for hit@3 / ACR@3 (每方法).
输出 _cache/e85_wadi_ext.json
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
from graphs.graph_ops import graph_features
from scripts.e1_anchor import perturb_graph, perturb_input, stable_seed
from scripts.e84_wadi import (load_attack, load_pool_rows, ATTACK_ROOTS,
                              segments_from_label, WIN, STRIDE, N_POOL,
                              CORR_TH, CAP_PER_SEG)
from scripts.e3_predictor import (phi_shape_feats, scorer_feats,
                                  consistency_feats, dev_feats)

CACHE = os.path.join(SRC, '_cache')


def boot_ci(vals, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, dtype=float)
    idx = rng.integers(0, len(vals), size=(B, len(vals)))
    bs = vals[idx].mean(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


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
        n_win = max(0, (b - a - WIN) // STRIDE + 1)
        take = min(n_win, CAP_PER_SEG)
        step = max(1, n_win // take) if n_win > take else 1
        cnt = 0
        for s in range(a, b - WIN + 1, STRIDE * step):
            if cnt >= take:
                break
            X_anom.append(Xa[s:s + WIN])
            R_anom.append(set(ridx))
            cnt += 1
    return np.stack(X_anom).astype(np.float32), R_anom, X_pool, Xp, starts


def main():
    X_anom, R_anom, X_pool, Xp, starts = build()
    n, d = X_anom.shape[0], X_anom.shape[2]
    print('windows', n, 'd', d, flush=True)

    scorer = make_scorer('iforest').fit(X_pool)
    s_pool = scorer.score(X_pool)
    ctx0 = Context(X_pool, scorer=scorer)
    mad = ctx0.pool_stats['mad']

    flat = Xp[starts[0]:starts[-1] + WIN]
    C = np.corrcoef(flat.T)
    A = (np.abs(C) > CORR_TH).astype(float)
    np.fill_diagonal(A, 0)
    gfeats = graph_features(A)

    rows = []
    results = {'methods': {}, 'cert': {}, 'zeroshot': None}

    def proc(mname, fn, graph, ctx_use=None):
        ctx = ctx_use if ctx_use is not None else ctx0
        phi_b = fn(X_anom, ctx.with_graph(A)) if graph else fn(X_anom, ctx)
        phis = [phi_b]
        for mi in range(8):
            rng = np.random.default_rng(stable_seed('e85', mname, mi))
            if graph:
                A_m = perturb_graph(A, ['del', 'add', 'rew'][mi % 3], 0.2, rng)
                phis.append(fn(X_anom, ctx.with_graph(A_m)))
            else:
                X_m = (X_anom + rng.normal(0, 1, X_anom.shape)
                       .astype(np.float32)
                       * (0.2 * mad)[None, None, :]).astype(np.float32)
                phis.append(fn(X_m, ctx))
        hits = M.hit_at_k(phi_b, R_anom, 3)
        cons = np.stack([M.acr_at_k(p, phi_b, 3) for p in phis[1:]]).mean(0)
        h3 = float(np.mean(hits))
        hlo, hhi = boot_ci(hits)
        a3 = float(np.mean(cons))
        alo, ahi = boot_ci(cons)
        results['methods'][mname] = dict(
            hit3=h3, hit3_ci=[hlo, hhi], acr3=a3, acr3_ci=[alo, ahi],
            sw=float(((cons > 0.8) & (hits == 0)).mean()))
        # 窗口特征行
        for i in range(n):
            pr = np.stack([p[i] for p in phis])
            rows.append(dict(
                dataset='wadi', data_kind='wadi', scorer='iforest',
                method=mname, graph_source='corr', window=i,
                label=int(hits[i]),
                **{f'g_{k2}': v for k2, v in gfeats.items()},
                **{f'c_{k2}': v for k2, v in consistency_feats(pr).items()},
                **{f'a_{k2}': v for k2, v in phi_shape_feats(phi_b[i]).items()},
                **{f's_{k2}': v for k2, v in scorer_feats(
                    float(scorer.score(X_anom[i:i + 1])[0]), s_pool).items()},
                **{f'd_{k2}': v for k2, v in dev_feats(X_anom[i], ctx0).items()}))
        # 证书 (e39 协议)
        D = np.stack(phis[1:]) - phi_b[None]
        delta = np.abs(D).max(0)                # (n, d) per-window max disp
        order = np.argsort(-phi_b, axis=1)
        cov = wrong = 0
        for i in range(n):
            gap = phi_b[i, order[i, 2]] - phi_b[i, order[i, 3]]
            eta = delta[i].max()
            if gap > 2 * eta:
                cov += 1
                wrong += int(hits[i] == 0)
        results['cert'][mname] = dict(n=n, n_certified=cov,
                                      certified_wrong=(wrong / cov) if cov else None)
        print(mname, results['methods'][mname],
              'cert', results['cert'][mname], flush=True)

    for mname, fn in GRAPH_DEPENDENT.items():
        proc(mname, fn, graph=True)
    for mname, fn in GRAPH_FREE.items():
        if mname == 'Grad':
            from scorers import TorchPCAScorer
            pca = make_scorer('pca').fit(X_pool)
            ctx_g = Context(X_pool, scorer=make_scorer('iforest').fit(X_pool))
            ctx_g.torch_scorer = TorchPCAScorer(pca)
            proc(mname, fn, graph=False, ctx_use=ctx_g)
        else:
            proc(mname, fn, graph=False)

    # 零样本: SCM 训练 -> WADI
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    dev_rows = json.load(open(os.path.join(CACHE, 'e3_windows.json'),
                              encoding='utf-8'))
    feats = [c for c in rows[0] if len(c) > 2 and c[0] in 'gcasd'
             and c[1] == '_']
    Xtr = np.array([[r.get(f, np.nan) for f in feats] for r in dev_rows])
    ytr = np.array([r['label'] for r in dev_rows])
    Xte = np.array([[r.get(f, np.nan) for f in feats] for r in rows])
    yte = np.array([r['label'] for r in rows])
    clf = HistGradientBoostingClassifier(random_state=0)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    results['zeroshot'] = dict(
        auroc=float(roc_auc_score(yte, p)),
        n_train=len(ytr), n_test=len(yte),
        label_prev=float(yte.mean()))
    # 每方法零样本
    per = {}
    for mname in set(r['method'] for r in rows):
        idx = [i for i, r in enumerate(rows) if r['method'] == mname]
        per[mname] = float(roc_auc_score(yte[idx], p[idx]))
    results['zeroshot']['per_method'] = per
    print('zero-shot SCM->WADI AUROC: %.3f (prev %.2f, per-method %s)'
          % (results['zeroshot']['auroc'], results['zeroshot']['label_prev'],
             {k: round(v, 2) for k, v in per.items()}), flush=True)

    out = os.path.join(CACHE, 'e85_wadi_ext.json')
    json.dump(results, open(out, 'w'), indent=1)
    rows_p = os.path.join(CACHE, 'e85_wadi_rows.json')
    json.dump(rows, open(rows_p, 'w'), default=float)
    print('->', out)


if __name__ == '__main__':
    main()
