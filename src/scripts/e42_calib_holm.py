# -*- coding: utf-8 -*-
"""E42: (a) 预测器概率校准 — LODO 出折概率的 ECE / Brier / 可靠性曲线;
      (b) GRAA 六个置换检验的 Holm 多重比较校正.
输出: _cache/e42_calibration.json
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from sklearn.ensemble import HistGradientBoostingClassifier

CACHE = os.path.join(SRC, '_cache')


def calibration():
    rows = json.load(open(os.path.join(CACHE, 'e3_windows.json'),
                          encoding='utf-8'))
    feats = [c for c in rows[0] if c[0] in 'gcasd' and c[1] == '_']
    X = np.array([[r.get(f, np.nan) for f in feats] for r in rows])
    y = np.array([r['label'] for r in rows])
    ds = np.array([r['dataset'] for r in rows])
    prob = np.full(len(rows), np.nan)
    for d in sorted(set(ds)):
        te, tr = ds == d, ds != d
        clf = HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                             learning_rate=0.08,
                                             random_state=0)
        clf.fit(X[tr], y[tr])
        prob[te] = clf.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(prob)
    p, yy = prob[ok], y[ok]
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    rel, cnt, acc = [], [], []
    for b in range(10):
        m = idx == b
        cnt.append(int(m.sum()))
        rel.append(float(p[m].mean()) if m.sum() else float('nan'))
        acc.append(float(yy[m].mean()) if m.sum() else float('nan'))
    ece = sum(c / c.sum() if False else c / len(p) * abs(a - r)
              for c, a, r in zip(cnt, acc, rel) if c > 0)
    ece = float(np.nansum([c / len(p) * abs(a - r) if c > 0 else 0
                           for c, a, r in zip(cnt, acc, rel)]))
    brier = float(np.mean((p - yy) ** 2))
    base = float(np.mean(yy))
    brier_skill = 1 - brier / (base * (1 - base))
    return dict(n=int(ok.sum()), prevalence=base, ece=ece, brier=brier,
                brier_skill=brier_skill, bin_count=cnt, bin_pred=rel,
                bin_true=acc)


def holm():
    tests = json.load(open(os.path.join(CACHE, 'e25_graa_signif.json')))['tests']
    items = sorted(tests.items(), key=lambda kv: kv[1]['perm_p'])
    m = len(items)
    adj, prev = [], 0.0
    for i, (k, v) in enumerate(items):
        a = min(1.0, max(prev, (m - i) * v['perm_p']))
        adj.append(a)
        prev = a
    return {k: dict(raw=v['perm_p'], holm=a) for (k, v), a in zip(items, adj)}


if __name__ == '__main__':
    out = dict(calibration=calibration(), holm=helm_holm() if False else holm())
    json.dump(out, open(os.path.join(CACHE, 'e42_calibration.json'), 'w'),
              default=float)
    c = out['calibration']
    print(f"calibration: ECE={c['ece']:.4f} Brier={c['brier']:.4f} "
          f"skill={c['brier_skill']:.3f} n={c['n']}")
    for k, v in out['holm'].items():
        print(f"  {k:34s} raw={v['raw']:.4f} holm={v['holm']:.4f}")
    print('-> e42_calibration.json')
