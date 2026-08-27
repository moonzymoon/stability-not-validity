# -*- coding: utf-8 -*-
"""E41: 每窗口推理开销 (毫秒/窗口) — 可部署性小节的数据.

测试床: SCM(medium, d=15) / TE(idv1_b0, d=41) / RCAEval(第一个可用单元, d≈46)
方法: zDev, AERec (graph-free); PropRank, DPTA-G (graph-dependent, 学到的图);
      GRAA(p=0.4) (含边排序+门控开销); 预测器 HGB 特征+推理.
计时: 50 窗口批次 x 3 次重复取中位数. 图为 PCMCI 学到的图 (部署场景).
输出: _cache/e41_runtime.json
"""
import os
import sys
import json
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.scm import make_sample, build_windows
from data.tep import load_tep_units
from data.rcaeval import load_ob_units
from graphs.sources import pcmci_graph
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graph_free import GRAPH_FREE
from attribution.graa_v4 import graa_v4_attribute

CACHE = os.path.join(SRC, '_cache')


def bench(fn, X, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(X)
        ts.append((time.perf_counter() - t0) / len(X) * 1000)
    return float(np.median(ts))


def get_units():
    s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2, seed=0,
                    noise_std=0.5, magnitude=0.5)
    w = build_windows(s)
    A = pcmci_graph(s['series'], tau_max=2)['adj']
    yield 'scm15', w, A, None
    u = load_tep_units()[0]
    d = np.load(os.path.join(CACHE, 'e24_graph_clean_te_idv1_b0.npz'))
    yield 'te41', u, d['A'], d['val']
    r = load_ob_units(reps=(1,))
    if r:
        ru = r[0]
        A_r = pcmci_graph(ru['series_normal'], tau_max=2)['adj']
        yield f'rca{ru["X_anom"].shape[2]}', ru, A_r.astype(float), None


def run():
    out = {}
    for name, u, A, val in get_units():
        Xa = u['X_anom'][:50]
        scorer = make_scorer('iforest').fit(u['X_pool'])
        ctx0 = Context(u['X_pool'], scorer=scorer)
        ctx_g = ctx0.with_graph(A)
        row = {}
        for mname, fn in (('zDev', GRAPH_FREE['zDev']),
                          ('AERec', GRAPH_FREE['AERec']),
                          ('PropRank', GRAPH_DEPENDENT['PropRank']),
                          ('DPTA-G', GRAPH_DEPENDENT['DPTA-G'])):
            row[mname] = bench(lambda X, f=fn, c=ctx_g: f(X, c), Xa)
        row['GRAA(p=0.4)'] = bench(
            lambda X: graa_v4_attribute(X, ctx_g, edge_conf=val, prune_frac=0.4)[0], Xa)
        out[name] = row
        print(name, {k: round(v, 2) for k, v in row.items()}, flush=True)

    # 预测器推理 (HGB, 特征构造 + predict)
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        rows = json.load(open(os.path.join(CACHE, 'e3_windows.json'),
                              encoding='utf-8'))
        feats = [c for c in rows[0] if c[0] in 'gcasd' and c[1] == '_']
        X = np.array([[r.get(f, np.nan) for f in feats] for r in rows[:2000]])
        y = np.array([r['label'] for r in rows[:2000]])
        clf = HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                             random_state=0).fit(X, y)
        out['predictor'] = dict(n_features=len(feats),
                                ms_per_window=bench(
                                    lambda Z: clf.predict_proba(Z), X[:50]))
        print('predictor', out['predictor'], flush=True)
    except Exception as e:
        print('predictor skip:', e)

    json.dump(out, open(os.path.join(CACHE, 'e41_runtime.json'), 'w'),
              default=float)
    print('-> e41_runtime.json')


if __name__ == '__main__':
    run()
