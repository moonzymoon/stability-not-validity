# -*- coding: utf-8 -*-
"""E78: 预测器一致性特征的 Mc 成本-精度曲线 (元宝⑤/DeepSeek残余).

与 e3 协议一致 (SCM 5seeds×2mag + TE, 3 打分器, 全方法), 一次算 6 个
扰动 phi, 输出前缀特征 c_cons_topk3@Mc (Mc=1,2,4,6), 各训 HistGB,
LODO AUROC 曲线。输出: _cache/e78_mc_curve.json
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

from scorers import make_scorer, TorchPCAScorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graph_free import GRAPH_FREE
from attribution.graa_v4 import graa_v4_attribute
from graphs.sources import pcmci_graph
from scripts.e1_anchor import perturb_graph, perturb_input, stable_seed
from scripts.e3_predictor import (collect_units, phi_shape_feats,
                                  dev_feats, scorer_feats)
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')
M_PERT = 6
MCS = (1, 2, 4, 6)


def run():
    t0 = time.time()
    rows = []
    units = collect_units()
    for u in units:
        graphs = u.get('graphs') or {
            'true': u['A_true'],
            'pcmci_clean': pcmci_graph(u['series_normal'])['adj'],
            'pcmci_anom': pcmci_graph(u['series_full'])['adj'],
        }
        for sk in ('iforest', 'pca', 'ocsvm'):
            scorer = make_scorer(sk).fit(u['X_pool'])
            s_pool = scorer.score(u['X_pool'])
            ctx0 = Context(u['X_pool'], scorer=scorer)
            ctx0.torch_scorer = TorchPCAScorer(scorer) if sk == 'pca' else None
            methods = list(GRAPH_DEPENDENT.items()) + list(GRAPH_FREE.items())
            for mname, fn in methods:
                if mname == 'Grad' and ctx0.torch_scorer is None:
                    continue
                graph_dep = mname in GRAPH_DEPENDENT
                for src_name, A in (graphs.items() if graph_dep
                                    else [('none', None)]):
                    ctx_b = ctx0.with_graph(A)
                    phi_b = fn(u['X_anom'], ctx_b)
                    hits = M.hit_at_k(phi_b, u['R_anom'], 3)
                    phi_perts = []
                    for mi in range(M_PERT):
                        rng = np.random.default_rng(stable_seed(
                            'cons', u['name'], sk, mname, src_name, mi))
                        if graph_dep:
                            fam = ['del', 'add', 'rew'][mi % 3]
                            A_m = perturb_graph(A, fam, 0.2, rng)
                            phi_perts.append(fn(u['X_anom'],
                                                ctx0.with_graph(A_m)))
                        else:
                            fam = ['noise', 'slice'][mi % 2]
                            X_m = perturb_input(u['X_anom'], fam,
                                                0.3 if fam == 'noise' else 2,
                                                rng, ctx0)
                            phi_perts.append(fn(X_m, ctx0))
                    # prefix consistency features
                    cfeat = {}
                    for mc in MCS:
                        ovs, t1s = [], []
                        for r in phi_perts[:mc]:
                            ovs.append(np.mean([
                                len(M.topk(r[i], 3) & M.topk(phi_b[i], 3)) / 3
                                for i in range(len(phi_b))]))
                        cfeat[mc] = float(np.mean(ovs))
                    for i in range(len(phi_b)):
                        rows.append(dict(
                            dataset=u['name'], scorer=sk, method=mname,
                            graph_source=src_name, window=i,
                            label=int(hits[i]),
                            **{f'c_mc{mc}': cfeat[mc] for mc in MCS},
                            **{f'a_{k}': v for k, v in
                               phi_shape_feats(phi_b[i]).items()},
                            **{f's_{k}': v for k, v in scorer_feats(
                                float(scorer.score(
                                    u['X_anom'][i:i + 1])[0]), s_pool).items()},
                            **{f'd_{k}': v for k, v in
                               dev_feats(u['X_anom'][i], ctx0).items()},
                        ))
        print(f'{u["name"]} done ({time.time()-t0:.0f}s, rows={len(rows)})',
              flush=True)
    json.dump(rows, open(os.path.join(CACHE, 'e78_mc_windows.json'), 'w'),
              default=float)

    # LODO per Mc
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    out = {}
    for mc in MCS:
        feats = [c for c in rows[0]
                 if c.startswith(('c_mc', 'a_', 's_', 'd_'))
                 and not c.startswith('c_mc')
                 or c == f'c_mc{mc}']
        X = np.array([[r.get(f, np.nan) for f in feats] for r in rows],
                     dtype=float)
        y = np.array([r['label'] for r in rows])
        ds = np.array([r['dataset'] for r in rows])
        aurocs = []
        for d in sorted(set(ds)):
            tr, te = ds != d, ds == d
            if y[te].std() == 0 or y[tr].std() == 0:
                continue
            clf = HistGradientBoostingClassifier(
                max_depth=4, max_iter=300, learning_rate=0.08,
                random_state=0).fit(X[tr], y[tr])
            aurocs.append(roc_auc_score(y[te],
                                        clf.predict_proba(X[te])[:, 1]))
        out[f'mc{mc}'] = round(float(np.mean(aurocs)), 3)
        print(f'Mc={mc}: LODO AUROC = {out[f"mc{mc}"]:.3f} ({len(aurocs)} folds)',
              flush=True)
    json.dump(out, open(os.path.join(CACHE, 'e78_mc_curve.json'), 'w'),
              indent=1)
    print('-> e78_mc_curve.json', f'total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    run()
