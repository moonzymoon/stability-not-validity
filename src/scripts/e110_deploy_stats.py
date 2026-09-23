# -*- coding: utf-8 -*-
"""E110: 部署统计硬化 (pre-submission audit).

对 E109 筛查与 E106b 过滤器补齐审稿人必问的统计:
  1) PR-AUC (average precision) — SWaT 零样本 / WADI 零样本 / WADI 200 重校准;
  2) top-30% / keep-20% 提升的显著性:
     - 超几何尾检验 (随机抽取同样数量窗口, 达到观测有效性的概率);
     - 窗口 bootstrap 95% CI;
     - 按 method x graph_source 聚类 bootstrap (保守);
  3) keep-20% 的 Wilson CI + 逐聚类 (method x graph) 符号一致率;
  4) 批量大小敏感性: 部署批 B in {25,50,100,200} 时 keep-20% 提升;
  5) 稳定错签名任务的单特征基线 AUROC (entropy / top-gap / hub / score-z).

协议与 E109 完全一致 (同 e3 训练, 同特征前缀 gcasd, 同 HGB 超参, random_state=0).
输出: _cache/e110_deploy_stats.json
"""
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
CACHE = os.path.join(SRC, '_cache')

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

PRED_PREFIX = 'gcasd'


def feats_labels(rows):
    feats = [c for c in rows[0]
             if len(c) > 2 and c[0] in PRED_PREFIX and c[1] == '_']
    X = np.array([[r.get(f, np.nan) for f in feats] for r in rows])
    y = np.array([r['label'] for r in rows])
    return X, y, feats


def hypergeom_tail(n_pop, k_good, n_draw, k_obs):
    """P(random n_draw-subset of n_pop contains >= k_obs good)."""
    from scipy.stats import hypergeom
    return float(hypergeom.sf(k_obs - 1, n_pop, k_good, n_draw))


def boot_ci(y, score, keep_frac, B=4000, seed=0, clusters=None):
    """Bootstrap CI of 'keep top keep_frac by score -> mean(y kept)'.

    clusters: optional array; resample clusters instead of windows.
    Difference-in-difference (kept minus rest) CI also returned.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    k = max(1, int(round(keep_frac * n)))
    order = np.argsort(-score, kind='stable')
    stats = []
    if clusters is None:
        for _ in range(B):
            idx = rng.integers(0, n, n)
            yy, ss = y[idx], score[idx]
            o = np.argsort(-ss, kind='stable')
            kk = max(1, int(round(keep_frac * n)))
            kept = yy[o[:kk]].mean()
            rest = yy[o[kk:]].mean() if n - kk > 0 else np.nan
            stats.append((kept, kept - rest))
    else:
        uniq = np.unique(clusters)
        for _ in range(B):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([np.where(clusters == c)[0] for c in pick])
            yy, ss = y[idx], score[idx]
            o = np.argsort(-ss, kind='stable')
            kk = max(1, int(round(keep_frac * len(idx))))
            kept = yy[o[:kk]].mean()
            rest = yy[o[kk:]].mean() if len(idx) - kk > 0 else np.nan
            stats.append((kept, kept - rest))
    stats = np.array(stats, dtype=float)
    stats = stats[~np.isnan(stats).any(axis=1)]
    lo, hi = np.percentile(stats[:, 0], [2.5, 97.5])
    dlo, dhi = np.percentile(stats[:, 1], [2.5, 97.5])
    return {'kept_mean': round(float(stats[:, 0].mean()), 4),
            'kept_ci95': [round(float(lo), 4), round(float(hi), 4)],
            'diff_ci95': [round(float(dlo), 4), round(float(dhi), 4)]}


def wilson(p, n, z=1.96):
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - hw), 4), round(min(1.0, c + hw), 4)]


def main():
    out = {}

    # ---------- 训练: 与 e109 完全一致 ----------
    e3 = json.load(open(os.path.join(CACHE, 'e3_windows.json'),
                        encoding='utf-8'))
    Xtr, ytr, feats = feats_labels(e3)
    clf = HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                         learning_rate=0.08, random_state=0)
    clf.fit(Xtr, ytr)

    # ---------- SWaT ----------
    sw = json.load(open(os.path.join(CACHE, 'e14_swat_pred_windows.json'),
                        encoding='utf-8'))
    Xsw, ysw, _ = feats_labels(sw)
    p = clf.predict_proba(Xsw)[:, 1]

    n = len(ysw)
    base = float(ysw.mean())
    order = np.argsort(-p, kind='stable')
    k30 = int(round(0.3 * n))
    top30 = float(ysw[order[:k30]].mean())
    out['swat_zeroshot'] = {
        'n': n, 'base': round(base, 4),
        'auroc': round(float(roc_auc_score(ysw, p)), 4),
        'prauc': round(float(average_precision_score(ysw, p)), 4),
        'prevalence': round(base, 4),
        'top30': round(top30, 4),
        'top30_hypergeom_p': hypergeom_tail(n, int(round(base * n)), k30,
                                            int(round(top30 * k30))),
        'top30_boot': boot_ci(ysw, p, 0.3, B=4000, seed=1),
        'top30_cluster_boot': boot_ci(
            ysw, p, 0.3, B=2000, seed=2,
            clusters=np.array([f"{r['method']}|{r['graph_source']}"
                               for r in sw])),
    }

    # 证书+过滤 (710 稳定窗内 keep-20%)
    m = np.array([r['c_cons_topk3'] >= 0.999 for r in sw])
    yc, pc = ysw[m], p[m]
    nc = int(m.sum())
    cb = float(yc.mean())
    o2 = np.argsort(-pc, kind='stable')
    k20 = int(round(0.2 * nc))
    keep20 = float(yc[o2[:k20]].mean())
    clusters_c = np.array([f"{r['method']}|{r['graph_source']}"
                           for r, mm in zip(sw, m) if mm])
    per_cluster = {}
    for c in np.unique(clusters_c):
        sel = clusters_c == c
        if sel.sum() >= 20:
            yy, ss = yc[sel], pc[sel]
            oo = np.argsort(-ss, kind='stable')
            kk = max(1, int(round(0.2 * sel.sum())))
            per_cluster[c] = {'n': int(sel.sum()),
                              'base': round(float(yy.mean()), 4),
                              'keep20': round(float(yy[oo[:kk]].mean()), 4)}
    signs = sum(1 for v in per_cluster.values() if v['keep20'] > v['base'])
    out['swat_certfilter'] = {
        'n_cert': nc, 'cert_base': round(cb, 4), 'keep20': round(keep20, 4),
        'keep20_wilson95': wilson(keep20, k20),
        'keep20_hypergeom_p': hypergeom_tail(nc, int(round(cb * nc)), k20,
                                             int(round(keep20 * k20))),
        'keep20_cluster_boot': boot_ci(yc, pc, 0.2, B=2000, seed=3,
                                       clusters=clusters_c),
        'per_cluster': per_cluster,
        'sign_consistency': f"{signs}/{len(per_cluster)}",
    }

    # 批量敏感性: 批内 keep-20% (模拟部署批)
    rng = np.random.default_rng(7)
    batch_stats = {}
    for Bsz in (25, 50, 100, 200):
        lifts, base_rate = [], []
        for _ in range(500):
            idx = rng.choice(nc, size=min(Bsz, nc), replace=False)
            yy, ss = yc[idx], pc[idx]
            oo = np.argsort(-ss, kind='stable')
            kk = max(1, int(round(0.2 * len(idx))))
            lifts.append(float(yy[oo[:kk]].mean()))
            base_rate.append(float(yy.mean()))
        batch_stats[f'B{Bsz}'] = {
            'mean_keep20': round(float(np.mean(lifts)), 4),
            'p5_keep20': round(float(np.percentile(lifts, 5)), 4),
            'mean_batch_base': round(float(np.mean(base_rate)), 4),
            'frac_above_base': round(float(np.mean(
                [l > b for l, b in zip(lifts, base_rate)])), 4)}
    out['swat_certfilter']['batch_sensitivity'] = batch_stats

    # ---------- WADI ----------
    wd = json.load(open(os.path.join(CACHE, 'e85_wadi_rows.json'),
                        encoding='utf-8'))
    Xwd, ywd, _ = feats_labels(wd)
    p0 = clf.predict_proba(Xwd)[:, 1]
    nw = len(ywd)
    bw = float(ywd.mean())
    ow = np.argsort(-p0, kind='stable')
    kw30 = int(round(0.3 * nw))
    t30w = float(ywd[ow[:kw30]].mean())
    out['wadi_zeroshot'] = {
        'n': nw, 'base': round(bw, 4),
        'auroc': round(float(roc_auc_score(ywd, p0)), 4),
        'prauc': round(float(average_precision_score(ywd, p0)), 4),
        'top30': round(t30w, 4),
        'top30_hypergeom_p': hypergeom_tail(nw, int(round(bw * nw)), kw30,
                                            int(round(t30w * kw30))),
        'top30_boot': boot_ci(ywd, p0, 0.3, B=4000, seed=4),
    }

    # WADI 200 重校准 (5 seeded splits, 同 e109 rng)
    rng = np.random.default_rng(19)
    rec = []
    for s in range(5):
        idx = rng.permutation(nw)
        lab = idx[:200]
        Xr = np.vstack([Xtr, Xwd[lab]])
        yr = np.concatenate([ytr, ywd[lab]])
        c2 = HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                            learning_rate=0.08,
                                            random_state=0)
        c2.fit(Xr, yr)
        rest = idx[200:]
        pr = c2.predict_proba(Xwd[rest])[:, 1]
        orr = np.argsort(-pr, kind='stable')
        k3 = int(round(0.3 * len(rest)))
        rec.append({'auroc': float(roc_auc_score(ywd[rest], pr)),
                    'prauc': float(average_precision_score(ywd[rest], pr)),
                    'base': float(ywd[rest].mean()),
                    'top30': float(ywd[rest][orr[:k3]].mean()),
                    'top30_hypergeom_p': hypergeom_tail(
                        len(rest), int(round(ywd[rest].mean() * len(rest))),
                        k3, int(round(
                            ywd[rest][orr[:k3]].mean() * k3)))})
    out['wadi_recalib200'] = {
        'prauc_mean': round(float(np.mean([r['prauc'] for r in rec])), 4),
        'top30_mean': round(float(np.mean([r['top30'] for r in rec])), 4),
        'top30_min': round(float(np.min([r['top30'] for r in rec])), 4),
        'top30_hypergeom_p_max': float(np.max(
            [r['top30_hypergeom_p'] for r in rec])),
        'auroc_mean': round(float(np.mean([r['auroc'] for r in rec])), 4)}

    # ---------- 稳定错签名: 单特征基线 ----------
    e9 = json.load(open(os.path.join(CACHE, 'e9_relation_windows.json'),
                        encoding='utf-8'))
    stable = [r for r in e3 + e9 if r['c_cons_topk3'] >= 0.999]
    ysg = np.array([1 - r['label'] for r in stable])  # 1 = wrong
    single = {}
    for f in ('a_attr_entropy', 'a_top_gap', 'a_top1_share',
              'g_indeg_max', 'g_indeg_entropy', 's_score_z', 'c_cons_top1'):
        v = np.array([r.get(f, np.nan) for r in stable], dtype=float)
        ok = ~np.isnan(v)
        auc = roc_auc_score(ysg[ok], v[ok])
        single[f] = {'auc': round(float(max(auc, 1 - auc)), 4),
                     'direction': 'higher→wrong' if auc >= 0.5 else
                     'lower→wrong'}
    out['stablewrong_single_feature'] = {
        'n': len(stable), 'base_wrong': round(float(ysg.mean()), 4),
        'single_feature_auc': single}

    path = os.path.join(CACHE, 'e110_deploy_stats.json')
    json.dump(out, io.open(path, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print('written', path)


if __name__ == '__main__':
    main()
