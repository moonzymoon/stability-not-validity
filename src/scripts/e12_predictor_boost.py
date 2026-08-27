# -*- coding: utf-8 -*-
"""E12 (方案1/8c/8d): 保形选择性归因 + 逐方法预测器 + 时间漂移协议."""
import os
import sys
import json
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
CACHE = os.path.join(SRC, '_cache')

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

rows = json.load(open(os.path.join(CACHE, 'e3_windows.json'), encoding='utf-8'))
feats = [c for c in rows[0] if c[0] in 'gcasd' and c[1] == '_']
X = np.array([[r.get(f, np.nan) for f in feats] for r in rows])
y = np.array([r['label'] for r in rows])
ds = np.array([r['dataset'] for r in rows])
sc = np.array([r['scorer'] for r in rows])
meth = np.array([r['method'] for r in rows])
win = np.array([r['window'] for r in rows])


def lodo_probs(mask_extra=None):
    prob = np.full(len(rows), np.nan)
    for d in sorted(set(ds)):
        te = (ds == d)
        tr = ~te
        if mask_extra is not None:
            tr &= ~mask_extra
        if tr.sum() < 300 or y[tr].std() == 0 or y[te].std() == 0:
            continue
        clf = HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                             learning_rate=0.08, random_state=0)
        clf.fit(X[tr], y[tr])
        prob[te] = clf.predict_proba(X[te])[:, 1]
    return prob


# ========== 方案1: 保形选择性归义 (split conformal 风格风险控制) ==========
def p1_conformal():
    """目标: 浮出覆盖率 c 下, 保证浮出窗口的期望 hit@3 >= 1-alpha.
    校准: 在每个 held-out 数据集内的校准折上选阈值 tau = 该折预测分位,
    验证实测风险是否 <= alpha (LTT/conformal risk control 思想)."""
    out = []
    # 部署协议: 每个数据集(部署)内部对半分 —— 校准折选阈值, 测试折验风险.
    # 这与 Algorithm 1 的 "injected calibration split" 一致 (标签来自注入校准).
    for alpha in (0.3, 0.4, 0.5):
        risks, covers = [], []
        for d in sorted(set(ds)):
            ep = win // 10                     # episode id (窗口级相关单元)
            all_ep = sorted(set(ep[ds == d]))
            half = len(all_ep) // 2
            cal_ep, te_ep = set(all_ep[:half]), set(all_ep[half:])
            idx_d = np.where(ds == d)[0]
            cal = np.array([i for i in idx_d if ep[i] in cal_ep])
            te = np.array([i for i in idx_d if ep[i] in te_ep])
            if len(cal) < 60 or len(te) < 30 or y[cal].std() == 0 or y[te].std() == 0:
                continue
            clf = HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                                 learning_rate=0.08,
                                                 random_state=0)
            # 校准模型也按 LODO 精神: 用其他数据集+本折校准训练
            tr = np.where(ds != d)[0]
            tr = np.concatenate([tr, cal])
            try:
                clf.fit(X[tr], y[tr])
            except Exception:
                continue
            p_te = clf.predict_proba(X[te])[:, 1]
            p_cal = clf.predict_proba(X[cal])[:, 1]
            tau = np.quantile(p_cal, 1 - alpha)
            surfaced = p_te >= tau
            risk = 1 - y[te][surfaced].mean() if surfaced.sum() else 1.0
            risks.append(risk)
            covers.append(surfaced.mean())
        out.append(dict(alpha=alpha, mean_risk=float(np.mean(risks)),
                        mean_coverage=float(np.mean(covers)),
                        risk_violation_rate=float(np.mean(
                            [r > alpha + 0.02 for r in risks]))))
        print(f'  alpha={alpha}: risk={np.mean(risks):.3f} '
              f'coverage={np.mean(covers):.2f} '
              f'violation={out[-1]["risk_violation_rate"]:.2f} '
              f'(n={len(risks)})')
    json.dump(out, open(os.path.join(CACHE, 'e12_p1.json'), 'w'))
    print('-> e12_p1.json')


# ========== 方案8c: 逐方法 (条件) 预测器 ==========
def p8c_per_method():
    out = {}
    for m in sorted(set(meth)):
        idx = meth == m
        if y[idx].std() == 0:
            continue
        prob = np.full(idx.sum(), np.nan)
        Xs, ys, dss = X[idx], y[idx], ds[idx]
        for d in sorted(set(dss)):
            te, tr = dss == d, dss != d
            if tr.sum() < 300 or te.sum() < 30 or ys[tr].std() == 0 or ys[te].std() == 0:
                continue
            try:
                clf = HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                                     learning_rate=0.08,
                                                     random_state=0)
                clf.fit(Xs[tr], ys[tr])
                prob[te] = clf.predict_proba(Xs[te])[:, 1]
            except Exception:
                continue
        ok = ~np.isnan(prob)
        if ok.sum() > 50 and ys[ok].std() > 0:
            try:
                out[m] = float(roc_auc_score(ys[ok], prob[ok]))
            except Exception as e:
                print(f'  [{m}] skip: {e}')
    print('per-method LODO AUROC:', {k: round(v, 3) for k, v in out.items()})
    json.dump(out, open(os.path.join(CACHE, 'e12_p8c.json'), 'w'))


# ========== 方案8d: 时间漂移协议 ==========
def p8d_temporal():
    """每数据集内: 前半窗口训练 -> 后半窗口测试 (模拟部署漂移)."""
    prob = np.full(len(rows), np.nan)
    for d in sorted(set(ds)):
        idx = np.where(ds == d)[0]
        idx = idx[np.argsort(win[idx])]
        half = len(idx) // 2
        tr, te = idx[:half], idx[half:]
        if len(tr) < 300 or len(te) < 30 or y[tr].std() == 0 or y[te].std() == 0:
            continue
        clf = HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                             learning_rate=0.08, random_state=0)
        clf.fit(X[tr], y[tr])
        prob[te] = clf.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(prob)
    auc = float(roc_auc_score(y[ok], prob[ok]))
    print(f'temporal-drift AUROC = {auc:.3f} (n={int(ok.sum())})')
    json.dump(dict(temporal_auroc=auc, n=int(ok.sum())),
              open(os.path.join(CACHE, 'e12_p8d.json'), 'w'))


# ========== RBO 附表数据 ==========
def p8_rbo_table():
    """ SCM 双根因场景的 RBO (deviational) — 从 E1 配置重算. """
    from data.scm import make_sample, build_windows
    from evaluation import metrics as Mmet
    from attribution.graph_dep import GRAPH_DEPENDENT
    from attribution import Context
    from scorers import make_scorer
    recs = []
    for seed in range(3):
        s = make_sample(n_nodes=15, T=5000, n_single=0, n_joint=7, seed=seed,
                        noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        A = s['adjacency'].astype(float)
        scorer = make_scorer('iforest').fit(w['X_pool'])
        ctx0 = Context(w['X_pool'], scorer=scorer)
        for mname, fn in GRAPH_DEPENDENT.items():
            phi = fn(w['X_anom'], ctx0.with_graph(A))
            recs.append(dict(seed=seed, method=mname,
                             rbo=float(Mmet.mean_rbo(phi, w['R_anom'])),
                             hit3=float(Mmet.mean_hit(phi, w['R_anom'], 3))))
    agg = collections.defaultdict(list)
    for r in recs:
        agg[r['method']].append((r['rbo'], r['hit3']))
    print('=== RBO (multi-root scenes) ===')
    for m, v in sorted(agg.items()):
        print(f'  {m:14s} RBO={np.mean([a for a, _ in v]):.3f} '
              f'hit@3={np.mean([b for _, b in v]):.3f}')
    json.dump({m: dict(rbo=float(np.mean([a for a, _ in v])),
                       hit3=float(np.mean([b for _, b in v])))
               for m, v in agg.items()},
              open(os.path.join(CACHE, 'e12_rbo.json'), 'w'))


if __name__ == '__main__':
    print('P1 conformal selective attribution:')
    p1_conformal()
    print('P8c per-method predictors:')
    p8c_per_method()
    print('P8d temporal drift:')
    p8d_temporal()
    print('RBO table:')
    p8_rbo_table()
