# -*- coding: utf-8 -*-
"""E14 (R14-C): SWaT 逐窗口CI + 预测器纳入SWaT (回应58条审查 条16/17/30/31/33/39)."""
import os
import sys
import json
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.swat import build_swat_units
from graphs.sources import pcmci_graph
from graphs import graph_ops as go
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT, _robust_z
from attribution.graph_free import GRAPH_FREE
from scripts.e8_round6 import gcn_rank_attribute, corr_graph
from evaluation import metrics as M
from scripts.e1_anchor import perturb_graph, stable_seed

CACHE = os.path.join(SRC, '_cache')


def boot_ci(vals, B=5000, seed=0):
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(B, len(vals)))
    bs = vals[idx].mean(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def part1_window_hits():
    """逐窗口 hit@3 + 事件bootstrap CI (每方法×图源)."""
    u = build_swat_units()
    A_silver = u['silver_adj']
    A_pcmci = pcmci_graph(u['series_normal'], tau_max=2)['adj']
    A_corr = corr_graph(u['series_normal'][:50000], thresh=0.3)
    graphs = {'silver': A_silver, 'pcmci': A_pcmci, 'corr': A_corr}
    for nm, A in graphs.items():
        r, p = go.edge_recall_precision(A, A_silver)
        dens = A.sum() / (51 * 50)
        print(f'graph[{nm}] E={int(A.sum())} rec={r:.2f} prec={p:.2f} dens={dens:.3f}',
              flush=True)
    out = []
    for sk in ('iforest', 'pca'):
        scorer = make_scorer(sk).fit(u['X_pool'])
        ctx0 = Context(u['X_pool'], scorer=scorer)
        for mname, fn in (list(GRAPH_DEPENDENT.items())
                          + [('GCN-Rank', gcn_rank_attribute)]):
            for gname, A in graphs.items():
                phi = fn(u['X_anom'], ctx0.with_graph(A))
                hits = M.hit_at_k(phi, u['R_anom'], 3)
                out.append(dict(scorer=sk, method=mname, graph_source=gname,
                                hits=[float(h) for h in hits]))
    json.dump(dict(graph_quality={nm: dict(edges=int(A.sum()),
                                           recall=float(go.edge_recall_precision(A, A_silver)[0]),
                                           precision=float(go.edge_recall_precision(A, A_silver)[1]),
                                           density=float(A.sum() / (51 * 50)))
                                  for nm, A in graphs.items()},
                   records=out),
              open(os.path.join(CACHE, 'e14_swat_windows.json'), 'w'))
    agg = collections.defaultdict(list)
    for r in out:
        agg[(r['method'], r['graph_source'])].extend(r['hits'])
    print('=== SWaT hit@3 with 95% CI (window bootstrap, n=480 pooled) ===')
    for k in (('PropRank', 'silver'), ('PropRank', 'pcmci'), ('PropRank', 'corr'),
              ('DPTA-G', 'silver'), ('DPTA-G', 'pcmci'), ('DPTA-G', 'corr'),
              ('GCN-Rank', 'silver'), ('GCN-Rank', 'pcmci')):
        v = agg[k]
        lo, hi = boot_ci(v)
        print(f'  {str(k):28s} {np.mean(v):.3f} [{lo:.3f},{hi:.3f}] n={len(v)}')
    return u, graphs


def part2_predictor(u, graphs):
    """SWaT 窗口级特征样本 + (a) LODO 含 SWaT; (b) zero-shot SCM->SWaT."""
    from scripts.e3_predictor import (phi_shape_feats, scorer_feats,
                                      consistency_feats, dev_feats)
    from graphs.graph_ops import graph_features
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    rows = []
    M = 6
    for sk in ('iforest', 'pca'):
        scorer = make_scorer(sk).fit(u['X_pool'])
        s_pool = scorer.score(u['X_pool'])
        ctx0 = Context(u['X_pool'], scorer=scorer)
        for mname, fn in list(GRAPH_DEPENDENT.items()):
            for gname, A in graphs.items():
                ctx_b = ctx0.with_graph(A)
                phi_b = fn(u['X_anom'], ctx_b)
                hits = M_hit = None
                from evaluation import metrics as Mm
                hits = Mm.hit_at_k(phi_b, u['R_anom'], 3)
                gfeats = graph_features(A)
                phis = [phi_b]
                for mi in range(M):
                    rng = np.random.default_rng(stable_seed(
                        'e14c', sk, mname, gname, mi))
                    A_m = perturb_graph(A, ['del', 'add', 'rew'][mi % 3], 0.2, rng)
                    phis.append(fn(u['X_anom'], ctx0.with_graph(A_m)))
                for i in range(len(phi_b)):
                    pr = np.stack([p[i] for p in phis])
                    rows.append(dict(
                        dataset='swat_inj', data_kind='swat', scorer=sk,
                        method=mname, graph_source=gname, window=i,
                        label=int(hits[i]),
                        **{f'g_{k2}': v for k2, v in gfeats.items()},
                        **{f'c_{k2}': v for k2, v in consistency_feats(pr).items()},
                        **{f'a_{k2}': v for k2, v in phi_shape_feats(phi_b[i]).items()},
                        **{f's_{k2}': v for k2, v in scorer_feats(
                            float(scorer.score(u['X_anom'][i:i + 1])[0]), s_pool).items()},
                        **{f'd_{k2}': v for k2, v in dev_feats(u['X_anom'][i], ctx0).items()}))
        print(f'[e14-pred] {sk} done ({len(rows)} rows)', flush=True)
    json.dump(rows, open(os.path.join(CACHE, 'e14_swat_pred_windows.json'), 'w'),
              default=float)

    dev_rows = json.load(open(os.path.join(CACHE, 'e3_windows.json'),
                              encoding='utf-8'))
    feats = [c for c in rows[0] if c[0] in 'gcasd' and c[1] == '_']

    def fit_eval(train_rows, test_rows):
        Xtr = np.array([[r.get(f, np.nan) for f in feats] for r in train_rows])
        ytr = np.array([r['label'] for r in train_rows])
        Xte = np.array([[r.get(f, np.nan) for f in feats] for r in test_rows])
        yte = np.array([r['label'] for r in test_rows])
        if ytr.std() == 0 or yte.std() == 0:
            return None
        clf = HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                             learning_rate=0.08, random_state=0)
        clf.fit(Xtr, ytr)
        return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))

    out = {}
    # (a) 全量 LODO (SCM+TE+SWaT), 只报 SWaT held-out
    aucs = [fit_eval([r for r in dev_rows + rows if r['dataset'] != 'swat_inj'],
                     [r for r in rows])]
    out['lodo_swat_held'] = aucs[0]
    print('LODO (SWaT held out):', aucs[0])
    # (b) zero-shot SCM -> SWaT
    out['zeroshot_scm_to_swat'] = fit_eval(dev_rows, rows)
    print('zero-shot SCM->SWaT:', out['zeroshot_scm_to_swat'])
    # (c) TE -> SWaT
    te_rows = [r for r in dev_rows if r['data_kind'] == 'te']
    out['zeroshot_te_to_swat'] = fit_eval(te_rows, rows)
    print('zero-shot TE->SWaT:', out['zeroshot_te_to_swat'])
    json.dump(out, open(os.path.join(CACHE, 'e14_pred_swat.json'), 'w'))


if __name__ == '__main__':
    u, graphs = part1_window_hits()
    part2_predictor(u, graphs)
