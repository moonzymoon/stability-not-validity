# -*- coding: utf-8 -*-
"""E15: Round-15 分析 (M60 保形随机分割 / M61 验证边重叠 / M62 证书CI / m59 象限CI)."""
import json
import os
import sys
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
CACHE = os.path.join(SRC, '_cache')

from sklearn.ensemble import HistGradientBoostingClassifier

rows = json.load(open(os.path.join(CACHE, 'e3_windows.json'), encoding='utf-8'))
feats = [c for c in rows[0] if c[0] in 'gcasd' and c[1] == '_']
X = np.array([[r.get(f, np.nan) for f in feats] for r in rows])
y = np.array([r['label'] for r in rows])
ds = np.array([r['dataset'] for r in rows])
win = np.array([r['window'] for r in rows])
ep = win // 10

out = {}

# ===== M60: 保形 — 随机 (可交换) episode 分割 =====
def conformal(split_mode):
    res = {}
    for alpha in (0.3, 0.4, 0.5):
        risks, covers = [], []
        rng = np.random.default_rng(0)
        for d in sorted(set(ds)):
            idx_d = np.where(ds == d)[0]
            eps = sorted(set(ep[idx_d]))
            if split_mode == 'random':
                perm = rng.permutation(eps)
                half = len(perm) // 2
                cal_ep, te_ep = set(perm[:half]), set(perm[half:])
            else:  # temporal
                half = len(eps) // 2
                cal_ep, te_ep = set(eps[:half]), set(eps[half:])
            cal = np.array([i for i in idx_d if ep[i] in cal_ep])
            te = np.array([i for i in idx_d if ep[i] in te_ep])
            if len(cal) < 60 or len(te) < 30 or y[cal].std() == 0 or y[te].std() == 0:
                continue
            tr = np.where(ds != d)[0]
            tr = np.concatenate([tr, cal])
            try:
                clf = HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                                     learning_rate=0.08,
                                                     random_state=0)
                clf.fit(X[tr], y[tr])
            except Exception:
                continue
            p_cal = clf.predict_proba(X[cal])[:, 1]
            p_te = clf.predict_proba(X[te])[:, 1]
            tau = np.quantile(p_cal, 1 - alpha)
            surfaced = p_te >= tau
            risk = 1 - y[te][surfaced].mean() if surfaced.sum() else 1.0
            risks.append(risk)
            covers.append(surfaced.mean())
        res[alpha] = dict(mean_risk=float(np.mean(risks)),
                          coverage=float(np.mean(covers)),
                          violation=float(np.mean([r > alpha + 0.02 for r in risks])),
                          n=len(risks))
    return res

out['conformal_random'] = conformal('random')
print('=== M60 conformal, exchangeable (random-episode) split ===')
for a, v in out['conformal_random'].items():
    print(f'  alpha={a}: risk={v["mean_risk"]:.3f} cov={v["coverage"]:.2f} '
          f'violation={v["violation"]:.2f} (n={v["n"]})')

# ===== M61: 贪心删边 vs 真实假边重叠 =====
from data.scm import make_sample, build_windows
from graphs.sources import pcmci_graph
from graphs import graph_ops as go
from attribution.graph_dep import GRAPH_DEPENDENT
from evaluation import metrics as M

prec_g, prec_r, base_rate = [], [], []
for seed in range(5):
    s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2, seed=seed,
                    noise_std=0.5, magnitude=0.5)
    w = build_windows(s)
    A_true = s['adjacency'].astype(float)
    A_bad = pcmci_graph(s['series'])['adj']
    from scorers import make_scorer
    from attribution import Context
    scorer = make_scorer('iforest').fit(w['X_pool'])
    ctx0 = Context(w['X_pool'], scorer=scorer)
    fn = GRAPH_DEPENDENT['PropRank']
    phi_bad = fn(w['X_anom'], ctx0.with_graph(A_bad))
    edges = go.edges_from_adj(A_bad)
    true_edges = set(go.edges_from_adj(A_true))
    spurious = [e for e in edges if e not in true_edges]
    infl = {}
    for e in edges:
        A_e = A_bad.copy()
        A_e[e[0], e[1]] = 0.0
        phi_e = fn(w['X_anom'], ctx0.with_graph(A_e))
        infl[e] = 1.0 - float(M.acr_at_k(phi_e, phi_bad, 3).mean())
    ranked = [e for e, _ in sorted(infl.items(), key=lambda kv: -kv[1])]
    rng = np.random.default_rng(seed)
    for k in (3, 5, 12):
        g_top = set(ranked[:k])
        prec_g.append(len(g_top & set(spurious)) / k)
        r_idx = rng.choice(len(edges), size=k, replace=False)
        r_top = {edges[i] for i in r_idx}
        prec_r.append(len(r_top & set(spurious)) / k)
        base_rate.append(len(spurious) / len(edges))
out['verify_edge_precision'] = dict(
    greedy=float(np.mean(prec_g)), random=float(np.mean(prec_r)),
    spurious_rate=float(np.mean(base_rate)))
print('=== M61 edge-verification precision ===')
print(json.dumps(out['verify_edge_precision'], indent=1))

# ===== M62: 证书 36% 的 Wilson CI =====
p3 = json.load(open(os.path.join(CACHE, 'e11_p3.json')))
ct = [r['hit'] for r in p3 if r['method'] == 'DPTA-G' and r['certified']]
n, p = len(ct), float(np.mean(ct))
z = 1.96
denom = 1 + z * z / n
center = (p + z * z / (2 * n)) / denom
half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
out['cert_wilson'] = dict(n=n, hit=p, lo=center - half, hi=center + half,
                          wrong=1 - p, wrong_lo=1 - center - half,
                          wrong_hi=1 - center + half)
print('=== M62 certificate Wilson CI ===')
print(json.dumps(out['cert_wilson'], indent=1))

# ===== m59: 象限占比 CI (已有缓存口径复核) =====
e1 = json.load(open(os.path.join(CACHE, 'e1_anchor.json')))
cc = collections.defaultdict(list)
for r in e1:
    if r.get('method', '_').startswith('_') or r.get('family') is None:
        continue
    if r['family'] != 'none' and r.get('acr3_base') is not None:
        cc[(r['dataset'], r['scorer'], r['method'], r['family'],
            r['strength'], r['graph_source'])].append((r['acr3_base'], r['hit3']))
pts = np.array([(np.mean([a for a, _ in v]), np.mean([h for _, h in v]))
                for v in cc.values()])
rng = np.random.default_rng(0)
bs = [((pts[i][:, 0] > 0.8) & (pts[i][:, 1] < 0.4)).mean()
      for i in [rng.integers(0, len(pts), len(pts)) for _ in range(5000)]]
out['quadrant_ci'] = [float(np.percentile(bs, 2.5)),
                      float(np.percentile(bs, 97.5))]
print('=== m59 quadrant CI ===', out['quadrant_ci'])

json.dump(out, open(os.path.join(CACHE, 'e15_analysis.json'), 'w'), indent=1)
print('-> e15_analysis.json')
