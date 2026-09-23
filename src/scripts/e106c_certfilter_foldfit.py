# -*- coding: utf-8 -*-
"""E106c: E106b 的折内预处理修正版 (折外中位数填补/标准化泄漏).

协议其余与 E106b v2 逐字一致: c_cons>=0.999 经验完全稳定子集, 轨迹级 LODO
logreg, 折内分位归一的 risk-coverage, 逐折 keep20 对照, 分位校准.
数值块改为每折内拟合中位数/均值/方差 (镜像 e108b). 覆盖 e106b_certfilter.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
CACHE = os.path.join(SRC, '_cache')

from scripts.e105_stablewrong import NUM_FEATS, build_matrix, group_of


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    e3 = json.load(open(os.path.join(CACHE, 'e3_windows.json'),
                        encoding='utf-8'))
    e9 = json.load(open(os.path.join(CACHE, 'e9_relation_windows.json'),
                        encoding='utf-8'))
    for r in e3 + e9:
        r['_grp'] = group_of(r['dataset'])
    stable = [r for r in e3 + e9 if r['c_cons_topk3'] >= 0.999]
    y = np.array([1 - r['label'] for r in stable])
    groups = np.array([r['_grp'] for r in stable])

    X, feats = build_matrix(stable)
    n_num = len(NUM_FEATS)
    Xnum = X[:, :n_num].astype(float)
    missind = np.isnan(Xnum).any(axis=1).astype(float)[:, None]
    Xoh = X[:, n_num:]
    Xfix = np.column_stack([missind, Xoh])

    p_lr = np.zeros(len(stable))
    fold_auc = []
    for g in sorted(set(groups)):
        tr, te = groups != g, groups == g
        if y[te].std() == 0 or y[tr].std() == 0:
            continue
        med = np.nanmedian(Xnum[tr], axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        Ztr = np.where(np.isnan(Xnum[tr]), med, Xnum[tr])
        mu, sd = Ztr.mean(axis=0), Ztr.std(axis=0) + 1e-9
        Atr = np.column_stack([(Ztr - mu) / sd, Xfix[tr]])
        Ate = np.column_stack(
            [(np.where(np.isnan(Xnum[te]), med, Xnum[te]) - mu) / sd,
             Xfix[te]])
        lr = LogisticRegression(max_iter=4000, C=1.0)
        lr.fit(Atr, y[tr])
        p_lr[te] = lr.predict_proba(Ate)[:, 1]
        fold_auc.append((g, float(roc_auc_score(y[te], p_lr[te]))))

    out = {'n_certified': len(stable),
           'certified_wrong_rate_pooled': round(float(y.mean()), 4),
           'fold_auc_logreg': {g: round(a, 4) for g, a in fold_auc},
           'fold_auc_logreg_mean': round(float(np.mean(
               [a for _, a in fold_auc])), 4),
           '_protocol': ('fold-internal median imputation and z-scoring '
                         '(e106c); subset = c_cons_topk3>=0.999 empirical '
                         'perfectly stable windows')}

    # 折内分位归一 (部署口径, 与 v2 相同)
    pct = np.zeros(len(stable))
    for g in sorted(set(groups)):
        m = groups == g
        if m.sum() < 50:
            pct[m] = np.nan
            continue
        pct[m] = p_lr[m].argsort().argsort() / (m.sum() - 1)

    keep_grid = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    order = np.argsort(pct, kind='stable')
    curve = []
    for keep in keep_grid:
        k = max(1, int(round(keep * np.sum(~np.isnan(pct)))))
        keep_idx = order[:k]
        rej = order[k:]
        curve.append(dict(
            keep_frac=keep, remaining_n=int(k),
            wrong_rate=round(float(y[keep_idx].mean()), 4),
            rejected_wrong_rate=round(float(y[rej].mean()), 4)
            if len(rej) else None))
    out['risk_coverage_foldnorm'] = curve

    fold_table = []
    for g in sorted(set(groups)):
        m = groups == g
        if m.sum() < 50:
            continue
        pg, yg = p_lr[m], y[m]
        o = np.argsort(pg, kind='stable')
        k20 = max(1, int(0.2 * m.sum()))
        fold_table.append(dict(
            fold=g, n=int(m.sum()),
            fold_base_wrong=round(float(yg.mean()), 4),
            keep20_wrong=round(float(yg[o[:k20]].mean()), 4),
            reject10_wrong=round(float(yg[o[-max(1, int(0.1 * m.sum())):]]
                                       .mean()), 4)))
    out['per_fold_table'] = fold_table
    out['keep20_below_base_folds'] = f"{sum(t['keep20_wrong'] < t['fold_base_wrong'] for t in fold_table)}/{len(fold_table)}"
    out['reject10_above_base_folds'] = f"{sum(t['reject10_wrong'] > t['fold_base_wrong'] for t in fold_table)}/{len(fold_table)}"

    ok = ~np.isnan(pct)
    bins = np.minimum((pct[ok] * 10).astype(int), 9)
    cal = []
    for b in range(10):
        m = bins == b
        if m.sum() >= 50:
            cal.append(dict(decile=b, n=int(m.sum()),
                            pred_wrong_pctile=round((b + 0.5) / 10, 2),
                            obs_wrong=round(float(y[ok][m].mean()), 4)))
    out['calibration_deciles_foldnorm'] = cal

    path = os.path.join(CACHE, 'e106b_certfilter.json')
    json.dump(out, open(path, 'w'), ensure_ascii=False, indent=1,
              default=float)
    print('[E106c fold-fit] written', path)
    print('pooled wrong =', out['certified_wrong_rate_pooled'],
          '| within-cert fold-mean AUC =', out['fold_auc_logreg_mean'])
    for c in curve:
        print(f"  keep {c['keep_frac']:.1f}: wrong {c['wrong_rate']:.3f}"
              f"  rejected {c['rejected_wrong_rate']}")
    print('per-fold keep20<base:', out['keep20_below_base_folds'],
          '| reject10>base:', out['reject10_above_base_folds'])


if __name__ == '__main__':
    main()
