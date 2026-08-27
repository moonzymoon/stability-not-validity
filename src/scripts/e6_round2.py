# -*- coding: utf-8 -*-
"""E6: Round-2 补强实验 (M7 预测器关系型覆盖 / M10 跨域 / M8 CI)。

M7  关系型异常窗口: 用与 e3 相同的特征工程提取窗口级样本, 评估
    (a) 冻结的偏差型训练预测器 zero-shot; (b) 混合训练(偏差+关系)重评 LODO。
M10 跨域 AUROC: 仅 SCM 训练 -> TE 测试; 仅 TE 训练 -> SCM 测试。
M8  PropRank 关系型 ACR 的 bootstrap CI。
输出: _cache/e6_relation_windows.json / e6_crossdomain.json / e6_ci.json
"""
import os
import sys
import json
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from scripts.e5_review import make_relation_sample
from data.scm import make_sample, build_windows
from graphs.sources import pcmci_graph
from scorers import make_scorer, TorchPCAScorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graph_free import GRAPH_FREE
from evaluation import metrics as M
from scripts.e1_anchor import perturb_graph, perturb_input, stable_seed
from scripts.e3_predictor import (phi_shape_feats, scorer_feats,
                                  consistency_feats, dev_feats, M_PERTURB)
from graphs.graph_ops import graph_features

CACHE = os.path.join(SRC, '_cache')
MAX_WINDOWS = 30


def build_relation_windows():
    """关系型异常的窗口级样本 (特征与 e3 完全同构)."""
    rows = []
    for seed in range(5):
        s = make_relation_sample(seed=seed)
        w = build_windows(s)
        n_w = min(MAX_WINDOWS, len(w['X_anom']))
        sel = np.linspace(0, len(w['X_anom']) - 1, n_w).astype(int)
        X_anom = w['X_anom'][sel]
        R = [w['R_anom'][i] for i in sel]
        A_true = s['adjacency'].astype(float)
        A_clean = pcmci_graph(s['normal'])['adj']
        A_anom = pcmci_graph(s['series'])['adj']
        graphs = {'true': A_true, 'pcmci_clean': A_clean, 'pcmci_anom': A_anom}
        for sk in ('iforest', 'pca'):
            scorer = make_scorer(sk).fit(w['X_pool'])
            s_pool = scorer.score(w['X_pool'])
            ctx0 = Context(w['X_pool'], scorer=scorer)
            if sk == 'pca':
                ctx0.torch_scorer = TorchPCAScorer(scorer)
            for mname, fn in (list(GRAPH_DEPENDENT.items())
                              + list(GRAPH_FREE.items())):
                if mname == 'Grad' and sk != 'pca':
                    continue
                graph_dep = mname in GRAPH_DEPENDENT
                for src_name, A in (graphs.items() if graph_dep
                                    else [('none', None)]):
                    ctx_b = ctx0.with_graph(A)
                    phi_b = fn(X_anom, ctx_b)
                    hits = M.hit_at_k(phi_b, R, 3)
                    gfeats = graph_features(A) if graph_dep else {}
                    for i in range(len(phi_b)):
                        phi_rows = [phi_b[i]]
                        for mi in range(M_PERTURB):
                            rng = np.random.default_rng(stable_seed(
                                'cons6', seed, sk, mname, src_name, i, mi))
                            if graph_dep:
                                fam = ['del', 'add', 'rew'][mi % 3]
                                A_m = perturb_graph(A, fam, 0.2, rng)
                                phi_m = fn(X_anom[i:i + 1],
                                           ctx0.with_graph(A_m))
                            else:
                                fam = ['noise', 'slice'][mi % 2]
                                X_m = perturb_input(
                                    X_anom[i:i + 1], fam,
                                    0.3 if fam == 'noise' else 2, rng, ctx0)
                                phi_m = fn(X_m, ctx0)
                            phi_rows.append(phi_m[0])
                        rows.append(dict(
                            dataset=f'rel_scm{seed}', data_kind='relation',
                            scorer=sk, method=mname, graph_source=src_name,
                            window=i, label=int(hits[i]),
                            **{f'g_{k}': v for k, v in gfeats.items()},
                            **{f'c_{k}': v for k, v in consistency_feats(
                                np.stack(phi_rows)).items()},
                            **{f'a_{k}': v for k, v in phi_shape_feats(
                                phi_b[i]).items()},
                            **{f's_{k}': v for k, v in scorer_feats(
                                float(scorer.score(X_anom[i:i + 1])[0]),
                                s_pool).items()},
                            **{f'd_{k}': v for k, v in dev_feats(
                                X_anom[i], ctx0).items()},
                        ))
        print(f'[M7] rel_scm{seed}: {len(rows)} rows', flush=True)
    return rows


def _fit_prob(rows_train, rows_test, model='hgb'):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    feats = [c for c in rows_train[0] if c[0] in 'gcasd' and c[1] == '_']
    Xtr = np.array([[r.get(f, np.nan) for f in feats] for r in rows_train])
    ytr = np.array([r['label'] for r in rows_train])
    Xte = np.array([[r.get(f, np.nan) for f in feats] for r in rows_test])
    yte = np.array([r['label'] for r in rows_test])
    if ytr.std() == 0 or yte.std() == 0:
        return None, None, None
    clf = HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                         learning_rate=0.08, random_state=0)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(yte, p)), yte, p


def main():
    # ---- M7 ----
    rel_rows = build_relation_windows()
    json.dump(rel_rows, open(os.path.join(CACHE, 'e6_relation_windows.json'),
                             'w'), ensure_ascii=False, default=float)
    dev_rows = json.load(open(os.path.join(CACHE, 'e3_windows.json'),
                              encoding='utf-8'))
    out = {}
    # zero-shot: 偏差型训练 -> 关系型测试 (逐数据集取均值)
    aucs = [_fit_prob(dev_rows, rel_rows[i:i + 2000])[0]
            for i in range(0, len(rel_rows), 2000)]
    aucs = [a for a in aucs if a is not None]
    p_all, y_all, prob_all = _fit_prob(dev_rows, rel_rows)
    out['zeroshot_dev_to_rel'] = p_all
    # 方法级细分
    by_m = collections.defaultdict(list)
    for r in rel_rows:
        by_m[r['method']].append(r)
    out['zeroshot_by_method'] = {}
    for m, rs in by_m.items():
        a, _, _ = _fit_prob(dev_rows, rs)
        if a is not None:
            out['zeroshot_by_method'][m] = a
    # 混合训练: dev+rel 混合, LODO 在关系型数据集上
    mixed = dev_rows + rel_rows
    rel_ds = sorted({r['dataset'] for r in rel_rows})
    mix_aucs = []
    for d in rel_ds:
        tr = [r for r in mixed if r['dataset'] != d]
        te = [r for r in mixed if r['dataset'] == d]
        a, _, _ = _fit_prob(tr, te)
        if a is not None:
            mix_aucs.append(a)
    out['mixed_LODO_rel'] = float(np.mean(mix_aucs))
    print('M7:', json.dumps(out, indent=1))

    # ---- M10 跨域 ----
    scm_rows = [r for r in dev_rows if r['data_kind'] == 'scm']
    te_rows = [r for r in dev_rows if r['data_kind'] == 'te']
    a1, _, _ = _fit_prob(scm_rows, te_rows)
    a2, _, _ = _fit_prob(te_rows, scm_rows)
    out['cross_scm_to_te'] = a1
    out['cross_te_to_scm'] = a2
    print('M10 cross-domain: SCM->TE', a1, ' TE->SCM', a2)

    # ---- M8 CI ----
    e5r = json.load(open(os.path.join(CACHE, 'e5_relation.json'),
                         encoding='utf-8'))
    pr_cells = np.array([r['acr3'] for r in e5r if r['method'] == 'PropRank'
                         and r.get('acr3') is not None])
    rng = np.random.default_rng(0)
    bs = [np.mean(pr_cells[rng.integers(0, len(pr_cells), len(pr_cells))])
          for _ in range(5000)]
    out['m8_pr_rel_acr_ci'] = [float(np.percentile(bs, 2.5)),
                               float(np.percentile(bs, 97.5))]
    out['m8_pr_rel_acr_mean'] = float(np.mean(pr_cells))
    # m10: transfer CI
    e6t = json.load(open(os.path.join(CACHE, 'e5_transfer.json'),
                         encoding='utf-8'))
    by = collections.defaultdict(dict)
    for r in e6t:
        by[(r['method'], r['dataset'])][r['graph_from']] = r['hit3']
    diffs = []
    for (m, ds), v in by.items():
        if m == 'PropRank' and 'self' in v:
            diffs.append(v['self'] - np.mean([x for k, x in v.items()
                                              if k != 'self']))
    bs2 = [np.mean(np.array(diffs)[rng.integers(0, len(diffs), len(diffs))])
           for _ in range(5000)]
    out['m10_transfer_ci'] = [float(np.percentile(bs2, 2.5)),
                              float(np.percentile(bs2, 97.5))]
    out['m10_transfer_sign'] = f"{sum(1 for x in diffs if x > 0)}/{len(diffs)}"
    json.dump(out, open(os.path.join(CACHE, 'e6_crossdomain.json'), 'w'),
              default=float, indent=1)
    print('M8/m10:', json.dumps({k: out[k] for k in
                                 ('m8_pr_rel_acr_ci', 'm8_pr_rel_acr_mean',
                                  'm10_transfer_ci', 'm10_transfer_sign')}, indent=1))


if __name__ == '__main__':
    main()
