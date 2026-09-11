# -*- coding: utf-8 -*-
"""E57: 创新增量三件套 — B(经验紧包络) + A(证书×预测器级联) + D(K/M敏感性).

复用 e39 证书设置: SCM medium (magnitude 0.5), 5 seeds, PropRank/DPTA-G,
M=8 个扰动实例 (循环 del/add/rew, 强度 q=0.1), iforest scorer, true graph.
关键区别: M=8 按前4(估计半)/后4(评估半)划分 ——
  B: worst-case 证书 gap>2*max(est半); empirical 证书 gap>2*Q_q(est半), q∈{0.5,0.75};
     在评估半测真实 top3 翻转 (集合不等) 率与覆盖率.
  A: 用 e3_predictor 特征函数构造行, LODO 协议(训练集剔除 scm*_m0.5 层)训 HGB 得 r̂;
     cert-only / predictor-only / cascade 三策略的 coverage-validity 曲线.
  D: 同批扰动下窗口级 ACR@K, K∈{1,3,5} × M∈{2,4,8} 敏感性.
输出: _cache/e57_innovation.json
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.scm import make_sample, build_windows
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from evaluation import metrics as M
from scripts.e1_anchor import perturb_graph, stable_seed
from graphs.graph_ops import graph_features
from scripts.e3_predictor import (phi_shape_feats, scorer_feats,
                                  consistency_feats, dev_feats)

CACHE = os.path.join(SRC, '_cache')
Q = 0.1
M_INST = 8
N_EST = 4  # 前4估计 / 后4评估

FEATS = None  # 运行时从 e3 行取列序


def topk_sets(phi, K):
    return [set(np.argsort(-row)[:K].tolist()) for row in phi]


def run():
    from sklearn.ensemble import HistGradientBoostingClassifier

    rows_cert = []       # B: 每窗口证书记录
    rows_feat = []       # A: e3 同 schema 特征行
    km_sens = []         # D: K/M 敏感性

    for seed in range(5):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                        seed=seed, noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        A = s['adjacency'].astype(float)
        scorer = make_scorer('iforest').fit(w['X_pool'])
        s_pool = scorer.score(w['X_pool'])
        ctx0 = Context(w['X_pool'], scorer=scorer)
        gfeats = graph_features(A)

        for mname in ('PropRank', 'DPTA-G'):
            fn = GRAPH_DEPENDENT[mname]
            phi0 = fn(w['X_anom'], ctx0.with_graph(A))
            hits = M.hit_at_k(phi0, w['R_anom'], 3)
            phis = []
            for mi in range(M_INST):
                rng = np.random.default_rng(stable_seed('e57', seed,
                                                        mname, mi))
                A_m = perturb_graph(A, ['del', 'add', 'rew'][mi % 3], Q, rng)
                phis.append(fn(w['X_anom'], ctx0.with_graph(A_m)))
            phis = np.stack(phis)                      # (M, n, d)
            D = np.abs(phis - phi0[None])              # (M, n, d)
            eta = D.max(axis=2).T                      # (n, M) 每实例位移
            order = np.argsort(-phi0, axis=1)
            gap = phi0[np.arange(len(phi0)), order[:, 2]] - \
                phi0[np.arange(len(phi0)), order[:, 3]]

            sets0 = topk_sets(phi0, 3)
            flip_eval = np.zeros(len(phi0))
            for mi in range(N_EST, M_INST):
                sets_m = topk_sets(phis[mi], 3)
                flip_eval += np.array([a != b for a, b in zip(sets_m, sets0)])
            flip_eval /= (M_INST - N_EST)              # 评估半平均翻转率

            est = eta[:, :N_EST]
            cert_worst = gap > 2 * est.max(axis=1)
            cert_emp = {q: gap > 2 * np.quantile(est, q, axis=1)
                        for q in (0.5, 0.75)}

            for i in range(len(phi0)):
                rows_cert.append(dict(
                    seed=seed, method=mname, window=i,
                    label=int(hits[i]), gap=float(gap[i]),
                    eta_max_est=float(est[i].max()),
                    flip_eval=float(flip_eval[i]),
                    cert_worst=bool(cert_worst[i]),
                    cert_emp50=bool(cert_emp[0.5][i]),
                    cert_emp75=bool(cert_emp[0.75][i])))
                rows_feat.append(dict(
                    dataset=f'e57_scm{seed}', data_kind='scm',
                    scorer='iforest', method=mname,
                    graph_source='true', window=i, label=int(hits[i]),
                    **{f'g_{k}': v for k, v in gfeats.items()},
                    **{f'c_{k}': v for k, v in consistency_feats(
                        np.stack([phi0[i]] + [phis[m][i]
                                              for m in range(M_INST)])).items()},
                    **{f'a_{k}': v for k, v in
                       phi_shape_feats(phi0[i]).items()},
                    **{f's_{k}': v for k, v in scorer_feats(
                        float(scorer.score(w['X_anom'][i:i + 1])[0]),
                        s_pool).items()},
                    **{f'd_{k}': v for k, v in
                       dev_feats(w['X_anom'][i], ctx0).items()}))

            # D: K/M 敏感性
            for K in (1, 3, 5):
                s0 = topk_sets(phi0, K)
                for MM in (2, 4, 8):
                    vals = []
                    for mi in range(MM):
                        sm = topk_sets(phis[mi], K)
                        vals += [len(a & b) / K for a, b in zip(sm, s0)]
                    km_sens.append(dict(seed=seed, method=mname, K=K, M=MM,
                                        acr=float(np.mean(vals))))
        print(f'seed {seed} done', flush=True)

    # ---- B 汇总 ----
    B = {}
    for mname in ('PropRank', 'DPTA-G'):
        sub = [r for r in rows_cert if r['method'] == mname]
        base_flip = np.mean([r['flip_eval'] for r in sub])
        entry = {'n': len(sub), 'flip_rate_all': float(base_flip)}
        for cname in ('cert_worst', 'cert_emp50', 'cert_emp75'):
            cov = np.mean([r[cname] for r in sub])
            fl = [r['flip_eval'] for r in sub if r[cname]]
            entry[cname] = dict(coverage=float(cov),
                                flip_rate=float(np.mean(fl)) if fl else None,
                                n_certified=int(cov * len(sub)))
        B[mname] = entry

    # ---- A: LODO 剔层预测 + 级联 ----
    e3 = json.load(open(os.path.join(CACHE, 'e3_windows.json'),
                        encoding='utf-8'))
    global FEATS
    FEATS = [c for c in e3[0] if c[0] in 'gcasd' and c[1] == '_']
    train = [r for r in e3 if '_m0.5' not in r['dataset']]
    Xt = np.array([[r.get(f, np.nan) for f in FEATS] for r in train])
    yt = np.array([r['label'] for r in train])
    clf = HistGradientBoostingClassifier(random_state=0)
    clf.fit(Xt, yt)
    Xe = np.array([[r.get(f, np.nan) for f in FEATS] for r in rows_feat])
    r_hat = clf.predict_proba(Xe)[:, 1]

    A_out = {}
    for mname in ('PropRank', 'DPTA-G'):
        idx = [i for i, r in enumerate(rows_feat) if r['method'] == mname]
        y = np.array([rows_feat[i]['label'] for i in idx])
        r = r_hat[idx]
        c75 = np.array([rows_cert[i]['cert_emp75']
                        for i in range(len(rows_feat))
                        if rows_feat[i]['method'] == mname])
        curve_pred, curve_casc = [], []
        for tau in np.linspace(0.15, 0.9, 16):
            m = r >= tau
            if m.sum() < 5:
                continue
            curve_pred.append(dict(tau=float(tau), coverage=float(m.mean()),
                                   validity=float(y[m].mean())))
            mc = c75 | m
            if mc.sum() < 5:
                continue
            curve_casc.append(dict(tau=float(tau), coverage=float(mc.mean()),
                                   validity=float(y[mc].mean())))
        mm = c75
        cert_point = dict(coverage=float(mm.mean()),
                          validity=float(y[mm].mean()) if mm.sum() else None,
                          n=int(mm.sum()))
        A_out[mname] = dict(cert_only=cert_point, predictor=curve_pred,
                            cascade=curve_casc,
                            auroc=float(_auroc(r, y)))

    # ---- D 汇总 ----
    D_out = {}
    for mname in ('PropRank', 'DPTA-G'):
        D_out[mname] = {(rec['K'], rec['M']): rec['acr']
                        for rec in km_sens if rec['method'] == mname
                        and rec['seed'] == 0}  # placeholder, 下面重算
        D_out[mname] = {}
        for K in (1, 3, 5):
            for MM in (2, 4, 8):
                vals = [rec['acr'] for rec in km_sens
                        if rec['method'] == mname and rec['K'] == K
                        and rec['M'] == MM]
                D_out[mname][f'K{K}_M{MM}'] = float(np.mean(vals))

    out = dict(B_empirical_certificates=B, A_cascade=A_out,
               D_km_sensitivity=D_out, q=Q,
               n_windows_per_method=len(rows_cert) // 2)
    json.dump(out, open(os.path.join(CACHE, 'e57_innovation.json'), 'w'),
              default=float)
    print(json.dumps(out, indent=1, default=str)[:2500])


def _auroc(r, y):
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y, r)
    except ValueError:
        return None


if __name__ == '__main__':
    run()
