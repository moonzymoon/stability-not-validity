# -*- coding: utf-8 -*-
"""E9: Round-7 补强实验.

M21  stale 数字修复: 用 M17 修复后的关系型数据重建窗口样本,
     重评 zero-shot / 混训 (摘要与正文中的 0.56/0.60 必须更新).
M22  预测器=难度代理? 的循环攻击: within-difficulty AUROC
     (按注入幅度分箱 + 按窗口偏差特征分箱, 控制难度后残余判别力).
M24  TE × AE scorer 补跑 (打分器不对称).
输出: _cache/e9_zeroshot.json / e9_withindiff.json / e9_te_ae.json
"""
import os
import sys
import json
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from scripts.e6_round2 import build_relation_windows, _fit_prob
from data.tep import load_tep_units
from scorers import make_scorer
from scripts.e8_round6 import AEScorer, gcn_rank_attribute
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')


def m21_zeroshot():
    # 修复后的关系型窗口样本 (make_relation_sample 已含噪声修复)
    rel_rows = build_relation_windows()
    json.dump(rel_rows, open(os.path.join(CACHE, 'e9_relation_windows.json'),
                             'w'), ensure_ascii=False, default=float)
    dev_rows = json.load(open(os.path.join(CACHE, 'e3_windows.json'),
                              encoding='utf-8'))
    out = {}
    a, _, _ = _fit_prob(dev_rows, rel_rows)
    out['zeroshot_dev_to_rel'] = a
    mixed = dev_rows + rel_rows
    rel_ds = sorted({r['dataset'] for r in rel_rows})
    aucs = []
    for d in rel_ds:
        tr = [r for r in mixed if r['dataset'] != d]
        te = [r for r in mixed if r['dataset'] == d]
        v, _, _ = _fit_prob(tr, te)
        if v is not None:
            aucs.append(v)
    out['mixed_LODO_rel'] = float(np.mean(aucs))
    print('M21 (post-fix):', out)
    json.dump(out, open(os.path.join(CACHE, 'e9_zeroshot.json'), 'w'))
    return out


def m22_withindiff():
    """控制难度后的残余判别力: (a) 按注入幅度分箱; (b) 按偏差集中度分箱."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    rows = json.load(open(os.path.join(CACHE, 'e3_windows.json'),
                          encoding='utf-8'))
    feats = [c for c in rows[0] if c[0] in 'gcasd' and c[1] == '_']
    X = np.array([[r.get(f, np.nan) for f in feats] for r in rows])
    y = np.array([r['label'] for r in rows])
    ds = np.array([r['dataset'] for r in rows])
    mag = np.array([r['dataset'].split('_m')[-1] if '_m' in r['dataset']
                    else 'te' for r in rows])
    out = {}
    # (a) 全局 LODO 预测概率, 然后在幅度箱内评估
    prob = np.full(len(rows), np.nan)
    for d in sorted(set(ds)):
        te, tr = ds == d, ds != d
        if y[tr].std() == 0 or y[te].std() == 0:
            continue
        clf = HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                             learning_rate=0.08,
                                             random_state=0)
        clf.fit(X[tr], y[tr])
        prob[te] = clf.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(prob)
    out['global_LODO'] = float(roc_auc_score(y[ok], prob[ok]))
    # 幅度箱内
    by_mag = {}
    for g in ('0.25', '0.5', '1.5', 'te'):
        m = ok & (mag == g)
        if m.sum() > 50 and y[m].std() > 0:
            by_mag[g] = dict(auroc=float(roc_auc_score(y[m], prob[m])),
                             n=int(m.sum()),
                             pos=float(y[m].mean()))
    out['by_magnitude'] = by_mag
    # (b) 偏差集中度箱 (d_dev_top3 三分位) 内评估 —— 控制信号质量
    dt = np.array([r.get('d_dev_top3', np.nan) for r in rows])
    qs = np.nanquantile(dt[ok], [1/3, 2/3])
    by_bin = {}
    for b, name in (((ok) & (dt <= qs[0]), 'low-conc'),
                    ((ok) & (dt > qs[0]) & (dt <= qs[1]), 'mid-conc'),
                    ((ok) & (dt > qs[1]), 'high-conc')):
        if b.sum() > 50 and y[b].std() > 0:
            by_bin[name] = dict(auroc=float(roc_auc_score(y[b], prob[b])),
                                n=int(b.sum()), pos=float(y[b].mean()))
    out['by_dev_concentration'] = by_bin
    print('M22:', json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(CACHE, 'e9_withindiff.json'), 'w'))
    return out


def m24_te_ae():
    """TE × AE scorer: 深度打分器在真实数据上的部署场景."""
    records = []
    for u in load_tep_units():
        cache = os.path.join(CACHE, f"te_graph_{u['name']}.npz")
        A = np.load(cache, allow_pickle=True)['A']
        scorer = AEScorer(seed=0).fit(u['X_pool'])
        ctx0 = Context(u['X_pool'], scorer=scorer)
        for mname, fn in GRAPH_DEPENDENT.items():
            phi = fn(u['X_anom'], ctx0.with_graph(A))
            records.append(dict(dataset=u['name'], scorer='ae', method=mname,
                                hit3=float(M.mean_hit(phi, u['R_anom'], 3))))
        phi = gcn_rank_attribute(u['X_anom'], ctx0.with_graph(A))
        records.append(dict(dataset=u['name'], scorer='ae', method='GCN-Rank',
                            hit3=float(M.mean_hit(phi, u['R_anom'], 3))))
        print(f"[M24] {u['name']} done", flush=True)
    json.dump(records, open(os.path.join(CACHE, 'e9_te_ae.json'), 'w'),
              default=float)
    agg = collections.defaultdict(list)
    for r in records:
        agg[r['method']].append(r['hit3'])
    print('=== M24 TE x AE (vs classical-scorer means) ===')
    for m in sorted(agg):
        print(f'  {m:14s} hit3={np.mean(agg[m]):.3f}')
    return records


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', 'm21'):
        m21_zeroshot()
    if which in ('all', 'm22'):
        m22_withindiff()
    if which in ('all', 'm24'):
        m24_te_ae()
