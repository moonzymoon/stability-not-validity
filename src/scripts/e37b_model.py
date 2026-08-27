# -*- coding: utf-8 -*-
"""E37b: 训练边置信度模型 + SWaT 银标准评估 + 下游 GRAA 对比.

模型: GBM, 特征 [|val|,出度,入度,密度,stability] (full) 与
[|val|,出度,入度,密度] (lite, 用于无 stability 的跨域场景).
评估: (a) SCM LOMO 泛化 AUC vs |val| 基线;
      (b) SWaT 学图边 vs 银标准 的 AUC;
      (c) 下游: GRAA 用学习置信度 vs |val| 剪枝 (SWaT/TE/RCAEval).
输出: _cache/e37b_model.json
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from data.swat import build_swat_units, silver_adj, COLIDX
from data.tep import load_tep_units
from data.rcaeval import load_ob_units
from graphs.sources import pcmci_graph
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graa_v4 import graa_v4_attribute
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')
FULL = 5
LITE = 4


def feats_mat(recs, d):
    return np.array([r['feat'][:d] for r in recs], float)


def run():
    recs = json.load(open(os.path.join(CACHE, 'e37_edges.json')))
    mags = sorted({r['mag'] for r in recs})

    # ---- (a) SCM leave-one-magnitude-out ----
    res = {'lomo': {}}
    for d, name in ((FULL, 'full'), (LITE, 'lite')):
        aucs_m, aucs_v = [], []
        for m in mags:
            tr = [r for r in recs if r['mag'] != m]
            te = [r for r in recs if r['mag'] == m]
            clf = HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                                 learning_rate=0.08,
                                                 random_state=0)
            clf.fit(feats_mat(tr, d), [r['label'] for r in tr])
            p = clf.predict_proba(feats_mat(te, d))[:, 1]
            y = [r['label'] for r in te]
            if len(set(y)) > 1:
                aucs_m.append(roc_auc_score(y, p))
                aucs_v.append(roc_auc_score(
                    y, [r['feat'][0] for r in te]))
        res['lomo'][name] = dict(model=float(np.mean(aucs_m)),
                                 val_baseline=float(np.mean(aucs_v)))
        print(f'LOMO {name}: model AUC={np.mean(aucs_m):.3f} '
              f'|val| AUC={np.mean(aucs_v):.3f}')

    # 全量训练两个模型
    clf_full = HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                              learning_rate=0.08,
                                              random_state=0).fit(
        feats_mat(recs, FULL), [r['label'] for r in recs])
    clf_lite = HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                              learning_rate=0.08,
                                              random_state=0).fit(
        feats_mat(recs, LITE), [r['label'] for r in recs])

    # ---- (b) SWaT 边分类 (银标准标签) ----
    d = np.load(os.path.join(CACHE, 'swat_pcmci_graph.npz'))
    A_sw, val_sw = d['A'], d['val']
    val_sw = val_sw / (val_sw.max() + 1e-9)
    silver = silver_adj()
    rng = np.random.default_rng(1)
    stable_sw = np.zeros_like(A_sw, dtype=float)
    series = build_swat_units()['series_normal'][:10000]
    for k in range(4):
        idx = np.sort(rng.choice(len(series), len(series) // 2,
                                 replace=False))
        rk = pcmci_graph(series[idx])
        stable_sw += (rk['adj'] > 0)
    stable_sw /= 4
    dens = float(A_sw.mean())
    outd, ind = A_sw.sum(1), A_sw.sum(0)
    edges = []
    for i, j in zip(*np.where(A_sw > 0)):
        edges.append(dict(i=int(i), j=int(j),
                          feat=[float(val_sw[i, j]), float(outd[i]),
                                float(ind[j]), dens,
                                float(stable_sw[i, j])]))
    y_sw, p_full, p_val = [], [], []
    for e in edges:
        lab = bool(silver[e['i'], e['j']])
        y_sw.append(int(lab))
        p_full.append(float(clf_full.predict_proba(
            np.array([e['feat'][:FULL]]))[0, 1]))
        p_val.append(e['feat'][0])
    res['swat_edges'] = dict(
        n=len(y_sw), n_pos=int(np.sum(y_sw)),
        auc_learned=float(roc_auc_score(y_sw, p_full)),
        auc_val=float(roc_auc_score(y_sw, p_val)))
    print('SWaT edges:', res['swat_edges'])

    # 学习置信度矩阵
    conf_full = np.zeros_like(A_sw, dtype=float)
    for e, p in zip(edges, p_full):
        conf_full[e['i'], e['j']] = p

    # ---- (c) 下游: SWaT GRAA (学习 vs val 置信度) ----
    u = build_swat_units()
    scorer = make_scorer('iforest').fit(u['X_pool'])
    ctx0 = Context(u['X_pool'], scorer=scorer)
    res['downstream'] = {}
    for gname, A in (('silver', silver), ('pcmci', A_sw)):
        conf = conf_full if gname == 'pcmci' else np.ones_like(A)
        phi, _ = graa_v4_attribute(u['X_anom'], ctx0.with_graph(A),
                                   edge_conf=conf, prune_frac=0.4)
        h_learn = float(M.mean_hit(phi, u['R_anom'], 3))
        conf_v = val_sw if gname == 'pcmci' else np.ones_like(A)
        phi2, _ = graa_v4_attribute(u['X_anom'], ctx0.with_graph(A),
                                    edge_conf=conf_v, prune_frac=0.4)
        h_val = float(M.mean_hit(phi2, u['R_anom'], 3))
        res['downstream'][f'swat/{gname}'] = dict(learned=h_learn,
                                                  val=h_val)
        print(f'SWaT {gname}: learned-conf hit@3={h_learn:.3f} '
              f'val-conf={h_val:.3f}')

    # ---- (c2) TE / RCAEval 下游 (lite 模型, 缓存图) ----
    def lite_conf(A, val):
        c = np.zeros_like(A, dtype=float)
        outd, ind = A.sum(1), A.sum(0)
        dens = float(A.mean())
        ii, jj = np.where(A > 0)
        if len(ii) == 0:
            return c
        F = np.stack([val[ii, jj], outd[ii], ind[jj],
                      np.full(len(ii), dens)], 1)
        c[ii, jj] = clf_lite.predict_proba(F)[:, 1]
        return c

    for u2 in load_tep_units():
        A = np.load(os.path.join(
            CACHE, f"te_graph_{u2['name']}.npz"),
            allow_pickle=True)['A']
        val = np.abs(pcmci_graph(u2['series_normal'])['val_matrix']).max(2)
        val = val / (val.max() + 1e-9)
        # val-conf 路径用缓存的 e24 val
        p = os.path.join(CACHE, f"e24_graph_clean_{u2['name']}.npz")
        val24 = np.load(p)['val'] if os.path.exists(p) else val
        scorer = make_scorer('iforest').fit(u2['X_pool'])
        ctx = Context(u2['X_pool'], scorer=scorer).with_graph(A)
        phi, _ = graa_v4_attribute(u2['X_anom'], ctx, edge_conf=lite_conf(A, val24),
                                   prune_frac=0.4)
        h_learn = float(M.mean_hit(phi, u2['R_anom'], 3))
        phi2, _ = graa_v4_attribute(u2['X_anom'], ctx, edge_conf=val24,
                                    prune_frac=0.4)
        h_val = float(M.mean_hit(phi2, u2['R_anom'], 3))
        k = f'te/{u2["name"]}'
        res['downstream'][k] = dict(learned=h_learn, val=h_val)
    te_l = np.mean([v['learned'] for k, v in res['downstream'].items()
                    if k.startswith('te')])
    te_v = np.mean([v['val'] for k, v in res['downstream'].items()
                    if k.startswith('te')])
    print(f'TE mean: learned={te_l:.3f} val={te_v:.3f}')

    units = [x for x in load_ob_units(reps=(1, 2, 3))
             if x['fault'] in ('cpu', 'mem')]
    for u3 in units:
        vals = {}
        for gname, kind in (('clean', 'clean'), ('anom', 'anom')):
            dd = np.load(os.path.join(
                CACHE, f"e31_graph_{kind}_{u3['name']}.npz"))
            A, val = dd['A'], dd['val']
            scorer = make_scorer('iforest').fit(u3['X_pool'])
            ctx = Context(u3['X_pool'], scorer=scorer).with_graph(A)
            phi, _ = graa_v4_attribute(u3['X_anom'], ctx,
                                       edge_conf=lite_conf(A, val),
                                       prune_frac=0.4)
            vals[f'learned_{gname}'] = float(
                M.mean_hit(phi, u3['R_anom'], 3))
            phi2, _ = graa_v4_attribute(u3['X_anom'], ctx,
                                        edge_conf=val, prune_frac=0.4)
            vals[f'val_{gname}'] = float(
                M.mean_hit(phi2, u3['R_anom'], 3))
        res['downstream'][f'rcaeval/{u3["name"]}'] = vals
    ob = res['downstream']
    ob_l = np.mean([v['learned_clean'] for k, v in ob.items()
                    if k.startswith('rcaeval')])
    ob_v = np.mean([v['val_clean'] for k, v in ob.items()
                    if k.startswith('rcaeval')])
    ob_la = np.mean([v['learned_anom'] for k, v in ob.items()
                     if k.startswith('rcaeval')])
    ob_va = np.mean([v['val_anom'] for k, v in ob.items()
                     if k.startswith('rcaeval')])
    print(f'RCAEval mean: learned {ob_l:.3f}/{ob_la:.3f} '
          f'val {ob_v:.3f}/{ob_va:.3f} (clean/anom)')

    json.dump(res, open(os.path.join(CACHE, 'e37b_model.json'), 'w'),
              indent=1, default=float)
    print('saved e37b_model.json')


if __name__ == '__main__':
    run()
