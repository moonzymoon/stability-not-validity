# -*- coding: utf-8 -*-
"""E7: Round-5 补强实验.

M18  LOSO 泄漏修复: 原 LOSO 的 train/test 共享同数据集窗口 (不同 scorer 配置),
    AUROC 虚高。补无泄漏的 LODSO (leave-one-dataset×scorer-out): test=(d,s),
    train=其余 d'!=d 的全部样本 —— 无共享数据集。
M19  PropRank damping 敏感性 (0.3/0.5/0.7) × 图源: 超参结论稳健性。
M20  象限占比的 K 敏感性 (K=1/3/5, 阈值 ACR>K 阈值同步)。
输出: _cache/e7_losdo.json / e7_damping.json / e7_ksens.json
"""
import os
import sys
import json
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

CACHE = os.path.join(SRC, '_cache')


def m18_losdo():
    rows = json.load(open(os.path.join(CACHE, 'e3_windows.json'),
                          encoding='utf-8'))
    feats = [c for c in rows[0] if c[0] in 'gcasd' and c[1] == '_']
    X = np.array([[r.get(f, np.nan) for f in feats] for r in rows])
    y = np.array([r['label'] for r in rows])
    ds = np.array([r['dataset'] for r in rows])
    sc = np.array([r['scorer'] for r in rows])
    aucs = []
    for d in sorted(set(ds)):
        for s in sorted(set(sc)):
            te = (ds == d) & (sc == s)
            tr = ds != d                     # 关键: 训练侧无该数据集
            if y[tr].std() == 0 or y[te].std() == 0:
                continue
            clf = HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                                 learning_rate=0.08,
                                                 random_state=0)
            clf.fit(X[tr], y[tr])
            p = clf.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p))
    out = dict(losdo_mean=float(np.mean(aucs)), n=len(aucs),
               losdo_min=float(np.min(aucs)), losdo_max=float(np.max(aucs)))
    print('M18 LODSO:', out)
    json.dump(out, open(os.path.join(CACHE, 'e7_losdo.json'), 'w'))
    return out


def m19_damping():
    from data.scm import make_sample, build_windows
    from graphs.sources import pcmci_graph
    from scorers import make_scorer
    from attribution import Context
    from attribution.graph_dep import proprank_attribute
    from evaluation import metrics as M
    from scripts.e1_anchor import perturb_graph, stable_seed, M_PERTURB
    records = []
    for seed in range(3):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2, seed=seed,
                        noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        A_true = s['adjacency'].astype(float)
        A_anom = pcmci_graph(s['series'])['adj']
        scorer = make_scorer('iforest').fit(w['X_pool'])
        ctx0 = Context(w['X_pool'], scorer=scorer)
        for damp in (0.3, 0.5, 0.7):
            for sname, A in (('true', A_true), ('anom', A_anom)):
                phi = proprank_attribute(w['X_anom'], ctx0.with_graph(A),
                                         damping=damp)
                h3 = float(M.mean_hit(phi, w['R_anom'], 3))
                acrs = []
                phi0 = proprank_attribute(w['X_anom'], ctx0.with_graph(A_true),
                                          damping=damp)
                for mi in range(M_PERTURB):
                    rng = np.random.default_rng(stable_seed(
                        'm19', seed, damp, sname, mi))
                    A_m = perturb_graph(A, ['del', 'add', 'rew'][mi % 3],
                                        0.2, rng)
                    phi_m = proprank_attribute(w['X_anom'],
                                               ctx0.with_graph(A_m),
                                               damping=damp)
                    acrs.append(M.acr_at_k(phi_m, phi, 3).mean())
                records.append(dict(seed=seed, damping=damp, source=sname,
                                    hit3=h3, acr3=float(np.mean(acrs))))
    json.dump(records, open(os.path.join(CACHE, 'e7_damping.json'), 'w'),
              default=float)
    agg = collections.defaultdict(list)
    for r in records:
        agg[(r['damping'], r['source'])].append((r['hit3'], r['acr3']))
    print('M19 damping (damping, source): (hit3, acr3)')
    for k in sorted(agg):
        h = np.mean([a for a, _ in agg[k]])
        a = np.mean([b for _, b in agg[k]])
        print(f'  d={k[0]} src={k[1]:5s} hit={h:.3f} acr={a:.3f}')


def m20_ksens():
    recs = json.load(open(os.path.join(CACHE, 'e1_anchor.json'),
                          encoding='utf-8'))
    out = {}
    for K in (1, 3, 5):
        cc = collections.defaultdict(list)
        for r in recs:
            if r.get('method', '_').startswith('_') or r.get('family') is None:
                continue
            if r['family'] != 'none' and r.get('acr3_base') is not None:
                cc[(r['dataset'], r['scorer'], r['method'], r['family'],
                    r['strength'], r['graph_source'])].append(
                    (r['acr3_base'], r['hit3']))
        # ACR 用 K=3 存档; hit 有 hit1/3/5 —— K 敏感性用 hit@K 的象限占比
        # (ACR 的 K 敏感性需重算, 此处先报 hit 侧 + 声明)
        pts = [(np.mean([a for a, _ in v]), np.mean([h for _, h in v]))
               for v in cc.values()]
        arr = np.array(pts)
        out[f'hit_sw_acr3'] = float(np.mean((arr[:, 0] > 0.8) & (arr[:, 1] < 0.4)))
    # 真正的 K 敏感性: 用窗口级 phi 重算 ACR@1/@5 与 hit@1/@5 的象限
    # 快速版: 单 seed 3 方法 全 K
    from data.scm import make_sample, build_windows
    from graphs.sources import pcmci_graph
    from scorers import make_scorer
    from attribution import Context
    from attribution.graph_dep import GRAPH_DEPENDENT
    from evaluation import metrics as M
    from scripts.e1_anchor import perturb_graph, stable_seed
    res = collections.defaultdict(list)
    for seed in range(3):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2, seed=seed,
                        noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        A = s['adjacency'].astype(float)
        scorer = make_scorer('iforest').fit(w['X_pool'])
        ctx0 = Context(w['X_pool'], scorer=scorer)
        for mname, fn in GRAPH_DEPENDENT.items():
            phi0 = fn(w['X_anom'], ctx0.with_graph(A))
            rng = np.random.default_rng(0)
            A_m = perturb_graph(A, 'del', 0.3, rng)
            phi_m = fn(w['X_anom'], ctx0.with_graph(A_m))
            for K in (1, 3, 5):
                acr = float(M.acr_at_k(phi_m, phi0, K).mean())
                hit = float(M.mean_hit(phi0, w['R_anom'], K))
                res[K].append((acr, hit))
    print('M20 K-sensitivity (mean over 9 method-seed cells):')
    for K in (1, 3, 5):
        a = np.mean([x[0] for x in res[K]])
        h = np.mean([x[1] for x in res[K]])
        sw = float(np.mean([(x[0] > 0.8) and (x[1] < 0.4) for x in res[K]]))
        print(f'  K={K}: acr={a:.3f} hit={h:.3f} SW-share={sw:.2f}')
        out[f'K{K}'] = dict(acr=float(a), hit=float(h), sw=sw)
    json.dump(out, open(os.path.join(CACHE, 'e7_ksens.json'), 'w'),
              default=float)


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', 'm18'):
        m18_losdo()
    if which in ('all', 'm19'):
        m19_damping()
    if which in ('all', 'm20'):
        m20_ksens()
