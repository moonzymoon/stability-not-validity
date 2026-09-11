# -*- coding: utf-8 -*-
"""E67 (夜航 P1): 窗口级可靠性预测器 — SS 第五床验证 + 跨系统零样本迁移.

回应 Limitations 原话 "the predictor is not yet validated on RCAEval windows":
  (a) SS 内 LODO-by-service (留一服务): 窗口特征全部署可观测, 标签=注入服务
      hit@3 (服务级真值投影, 与四床口径一致)
  (b) SS 内 pooled 参考 (GroupKFold by unit)
  (c) OB-holdout 窗口表 (rep4-5, 17 cpu/mem 单元, 图缓存复用 e66)
  (d) 跨系统零样本: SS→OB-holdout 与 OB-holdout→SS (检验 "calibration does
      not transfer zero-shot" 是否跨系统成立)
  (e) 少量目标窗重校准: OB-holdout 训练 + 200 SS 窗 → 其余 SS 测试

特征/模型复用 e3_predictor 的函数与 schema (gcasd 五组, HistGB random_state=0),
保证与已发表协议可比。确定性: stable_seed 全程; AERCA/GPU 不涉及。
输出: _cache/e67_ss_windows.json, e67_obh_windows.json, e67_results.json
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
import sys
import json
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.rcaeval2 import load_units
from data.rcaeval import load_ob_units
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graph_free import GRAPH_FREE
from attribution.graa_v4 import graa_v4_attribute
from scripts.e1_anchor import perturb_graph, perturb_input, stable_seed
from scripts.e3_predictor import (phi_shape_feats, consistency_feats,
                                  dev_feats, scorer_feats, fit_predictor)
from graphs.sources import pcmci_graph
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')
M_PERTURB = 6
METHODS = ('PropRank', 'GRAA(p=0.4)', 'zDev')


def graphs_for(u, tag):
    out = {}
    for g in ('clean', 'anom'):
        path = os.path.join(CACHE, f'e66_graph_{g}_{u["name"]}.npz')
        if os.path.exists(path):
            d = np.load(path)
            out[f'pcmci_{g}'] = (d['A'], d['val'])
        else:
            # e66 守卫跳过的单元: 现算并落盘 (与 e66 同参数 tau_max=1)
            series = (u['series_normal'] if g == 'clean'
                      else u['series_full'])
            r = pcmci_graph(series, tau_max=1)
            val = np.abs(r['val_matrix']).max(2)
            val = val / (val.max() + 1e-9)
            np.savez(path, A=r['adj'], val=val)
            out[f'pcmci_{g}'] = (r['adj'], val)
    return out


def build_rows(units, tag):
    rows = []
    t0 = time.time()
    for ui, u in enumerate(units):
        graphs = graphs_for(u, tag)
        for sk in ('iforest', 'pca'):
            scorer = make_scorer(sk).fit(u['X_pool'])
            s_pool = scorer.score(u['X_pool'])
            ctx0 = Context(u['X_pool'], scorer=scorer)
            for mname in METHODS:
                graph_dep = mname in GRAPH_DEPENDENT or mname.startswith('GRAA')
                sources = (('pcmci_clean', 'pcmci_anom') if graph_dep
                           else ('none',))
                for src in sources:
                    if graph_dep:
                        A, val = graphs[src]
                        ctx_b = ctx0.with_graph(A)
                        if mname == 'GRAA(p=0.4)':
                            fn = lambda X, c: graa_v4_attribute(  # noqa: E731
                                X, c, edge_conf=val, prune_frac=0.4)[0]
                        else:
                            fn = GRAPH_DEPENDENT[mname]
                    else:
                        ctx_b = ctx0
                        fn = GRAPH_FREE[mname]
                    phi_b = fn(u['X_anom'], ctx_b)
                    hits = M.hit_at_k(phi_b, u['R_anom'], 3)
                    phi_perts = []
                    for mi in range(M_PERTURB):
                        rng = np.random.default_rng(stable_seed(
                            'e67', tag, u['name'], sk, mname, src, mi))
                        if graph_dep:
                            fam = ['del', 'add', 'rew'][mi % 3]
                            A_m = perturb_graph(A, fam, 0.2, rng)
                            cm = ctx0.with_graph(A_m)
                            phi_perts.append(fn(u['X_anom'], cm))
                        else:
                            fam = ['noise', 'slice'][mi % 2]
                            X_m = perturb_input(u['X_anom'], fam,
                                                0.3 if fam == 'noise' else 2,
                                                rng, ctx0)
                            phi_perts.append(fn(X_m, ctx0))
                    for i in range(len(phi_b)):
                        phi_rows = np.stack([phi_b[i]]
                                            + [pp[i] for pp in phi_perts])
                        rows.append(dict(
                            dataset=u['name'], service=u['service'],
                            data_kind=tag, scorer=sk, method=mname,
                            graph_source=src, window=i,
                            label=int(hits[i]),
                            **{f'c_{k}': v for k, v in
                               consistency_feats(phi_rows).items()},
                            **{f'a_{k}': v for k, v in
                               phi_shape_feats(phi_b[i]).items()},
                            **{f's_{k}': v for k, v in scorer_feats(
                                float(scorer.score(
                                    u['X_anom'][i:i + 1])[0]),
                                s_pool).items()},
                            **{f'd_{k}': v for k, v in
                               dev_feats(u['X_anom'][i], ctx0).items()},
                        ))
        print(f'[{tag} {ui+1}/{len(units)}] {u["name"]} '
              f'rows={len(rows)} ({time.time()-t0:.0f}s)', flush=True)
    return rows


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def main():
    t_all = time.time()
    ss_path = os.path.join(CACHE, 'e67_ss_windows.json')
    obh_path = os.path.join(CACHE, 'e67_obh_windows.json')
    if os.path.exists(ss_path):
        ss = json.load(open(ss_path))
    else:
        units = load_units('ss', reps=(1, 2, 3))
        n0 = len(units)
        units = [u for u in units
                 if not np.isnan(u['series_full']).any()
                 and not np.isnan(u['series_normal']).any()]
        print(f'ss units: {n0} -> {len(units)} (NaN dropped)', flush=True)
        ss = build_rows(units, 'ss')
        json.dump(ss, open(ss_path, 'w'), default=float)
    if os.path.exists(obh_path):
        obh = json.load(open(obh_path))
    else:
        units = [u for u in load_ob_units(reps=(4, 5))
                 if u['fault'] in ('cpu', 'mem')
                 and not np.isnan(u['series_full']).any()]
        obh = build_rows(units, 'obh')
        json.dump(obh, open(obh_path, 'w'), default=float)
    print(f'rows: ss={len(ss)} obh={len(obh)}', flush=True)

    results = []

    # (a) SS LODO by service
    from sklearn.ensemble import HistGradientBoostingClassifier
    services = sorted({r['service'] for r in ss})
    feats = [c for c in ss[0] if c[0] in 'gcasd' and c[1] == '_']
    X = np.array([[r.get(f, np.nan) for f in feats] for r in ss],
                 dtype=float)
    y = np.array([r['label'] for r in ss])
    for svc in services:
        tr = np.array([r['service'] != svc for r in ss])
        te = ~tr
        if y[te].std() == 0:
            continue
        clf = HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                             learning_rate=0.08,
                                             random_state=0).fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        results.append(dict(protocol='SS_LODO_service', held=svc,
                            auroc=auc(y[te], p), n_test=int(te.sum())))

    # (b) pooled reference: GroupKFold by unit (5 folds)
    from sklearn.model_selection import GroupKFold
    from sklearn.ensemble import HistGradientBoostingClassifier
    feats = [c for c in ss[0] if c[0] in 'gcasd' and c[1] == '_']
    X = np.array([[r.get(f, np.nan) for f in feats] for r in ss], dtype=float)
    y = np.array([r['label'] for r in ss])
    groups = np.array([r['dataset'] for r in ss])
    aucs = []
    for tr_i, te_i in GroupKFold(n_splits=5).split(X, y, groups):
        clf = HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                             learning_rate=0.08,
                                             random_state=0).fit(X[tr_i],
                                                                 y[tr_i])
        aucs.append(auc(y[te_i], clf.predict_proba(X[te_i])[:, 1]))
    results.append(dict(protocol='SS_pooled_GroupKFold5',
                        auroc=float(np.mean(aucs)),
                        auroc_per_fold=[float(a) for a in aucs]))

    # (d) cross-system zero-shot
    Xo = np.array([[r.get(f, np.nan) for f in feats] for r in obh],
                  dtype=float)
    yo = np.array([r['label'] for r in obh])
    clf_ss = HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                            learning_rate=0.08,
                                            random_state=0).fit(X, y)
    results.append(dict(protocol='zero-shot SS->OBh',
                        auroc=auc(yo, clf_ss.predict_proba(Xo)[:, 1])))
    clf_ob = HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                            learning_rate=0.08,
                                            random_state=0).fit(Xo, yo)
    results.append(dict(protocol='zero-shot OBh->SS',
                        auroc=auc(y, clf_ob.predict_proba(X)[:, 1])))

    # (e) recalibration: OBh + 200 SS windows -> remaining SS (by unit)
    rng = np.random.default_rng(stable_seed('e67', 'recal'))
    unit_names = sorted(set(groups))
    rng.shuffle(unit_names)
    cal_units = set()
    n = 0
    for un in unit_names:
        cal_units.add(un)
        n += sum(1 for r in ss if r['dataset'] == un)
        if n >= 200:
            break
    tr = np.array([r['dataset'] in cal_units for r in ss])
    tr_rows = [r for r in ss if r['dataset'] in cal_units] + obh
    Xtr = np.array([[r.get(f, np.nan) for f in feats] for r in tr_rows],
                   dtype=float)
    ytr = np.array([r['label'] for r in tr_rows])
    te = np.array([r['dataset'] not in cal_units for r in ss])
    clf = HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                         learning_rate=0.08,
                                         random_state=0).fit(Xtr, ytr)
    results.append(dict(protocol='recal OBh+200SS->SS', n_cal_units=len(cal_units),
                        auroc=auc(y[te], clf.predict_proba(X[te])[:, 1]),
                        n_test=int(te.sum())))

    out = dict(meta=dict(script='e67_predictor_ss',
                         rows_ss=len(ss), rows_obh=len(obh),
                         wall_s=round(time.time() - t_all, 1)),
               results=results)
    json.dump(out, open(os.path.join(CACHE, 'e67_results.json'), 'w'),
              default=float, indent=1)
    print('\n=== E67 predictor results ===')
    for r in results:
        print(f"  {r['protocol']:26} auroc={r['auroc']:.3f} "
              f"({r.get('n_test', '')})")
    print(f'total {time.time()-t_all:.0f}s')


if __name__ == '__main__':
    main()
