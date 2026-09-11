# -*- coding: utf-8 -*-
"""Mock-review 处置验证 v2: 元宝 P0③ 条件对比 + Q2 预测器部署态."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import sys
import json
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
CACHE = os.path.join(SRC, '_cache')


def cluster_boot_rd(wrong, is_hi, groups, n=2000, seed=20260908):
    """按 unit 聚类重采样, 组内算 P(wrong|hi)-P(wrong|lo)."""
    rd = float(np.mean(wrong[is_hi]) - np.mean(wrong[~is_hi]))
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    rds = []
    for _ in range(n):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in pick])
        w, h = wrong[idx], is_hi[idx]
        if h.sum() == 0 or (~h).sum() == 0:
            continue
        rds.append(w[h].mean() - w[~h].mean())
    return rd, float(np.percentile(rds, 2.5)), float(np.percentile(rds, 97.5))


def main():
    rows = json.load(open(os.path.join(CACHE, 'e3_windows.json')))
    print(f'window rows: {len(rows)}')
    out = {}

    for scope in ('all', 'true', 'pcmci_anom'):
        sub = [r for r in rows
               if (scope == 'all' or r['graph_source'] == scope)
               and r['method'] != 'Random']
        by_m = collections.defaultdict(lambda: ([], [], []))
        for r in sub:
            by_m[r['method']][0].append(1.0 - r['label'])
            by_m[r['method']][1].append(r['c_cons_topk3'] > 0.8)
            by_m[r['method']][2].append(r['dataset'])
        res = []
        for m, (wl, hl, gl) in sorted(by_m.items()):
            w, h, g = np.array(wl), np.array(hl), np.array(gl)
            if h.sum() < 10 or (~h).sum() < 10:
                continue
            rd, lo95, hi95 = cluster_boot_rd(w, h, g)
            res.append(dict(method=m,
                            p_wrong_high=round(float(w[h].mean()), 3),
                            p_wrong_low=round(float(w[~h].mean()), 3),
                            rd=round(rd, 3),
                            ci=[round(lo95, 3), round(hi95, 3)],
                            n_hi=int(h.sum()), n_lo=int((~h).sum())))
        out[f'conditional_{scope}'] = res
        print(f'\n=== (a) P(wrong|cons>0.8) vs P(wrong|cons<=0.8) [{scope}] ===')
        for r in res:
            print(f"  {r['method']:12} hi={r['p_wrong_high']:.3f} "
                  f"lo={r['p_wrong_low']:.3f} RD={r['rd']:+.3f} "
                  f"CI[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] "
                  f"(n_hi={r['n_hi']}, n_lo={r['n_lo']})")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    dep = [r for r in rows if r['graph_source'] == 'pcmci_anom']
    feats = [c for c in dep[0] if c[0] in 'gcasd' and c[1] == '_']
    X = np.array([[r.get(f, np.nan) for f in feats] for r in dep], dtype=float)
    y = np.array([r['label'] for r in dep])
    ds = np.array([r['dataset'] for r in dep])
    aurocs = []
    for d in sorted(set(ds)):
        tr, te = ds != d, ds == d
        if y[te].std() == 0 or y[tr].std() == 0:
            continue
        clf = HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                             learning_rate=0.08,
                                             random_state=0).fit(X[tr], y[tr])
        aurocs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    out['predictor_deployment'] = dict(
        n_rows=len(dep), lodo_auroc=[round(float(a), 3) for a in aurocs],
        mean=round(float(np.mean(aurocs)), 3))
    print(f"\n=== (b) 预测器@污染图部署态 (pcmci_anom rows) ===")
    print(f"  rows={len(dep)}, LODO AUROC={out['predictor_deployment']['mean']:.3f}")

    json.dump(out, open(os.path.join(CACHE, 'mock_p0_verify.json'), 'w'),
              indent=1)
    print('-> mock_p0_verify.json')


if __name__ == '__main__':
    main()
