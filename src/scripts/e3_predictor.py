# -*- coding: utf-8 -*-
"""E3 归因可信度预测器: 窗口级 hit@3 预测 (v2 必改3 全套)。

特征 (全部部署时可观测, 无需测试标签):
  图特征     graph_features(A_base): 密度/入度熵/入度极值/汇点占比...
  一致性特征  窗口级扰动一致性: phi 在 M 个扰动图/输入下的 TopK 平均重叠
             (部署时可通过多扰动获得, 是 ACR 的窗口级等价物) + top1 稳定份额
  归因特征    phi 分布形状: top1 份额 / 归一熵 / top-gap / top3 份额
  打分器特征  score-margin (窗口分 vs 池分位) / 分数分布熵 —— 消融对象
  数据特征    窗口内偏差熵 / 偏差集中度 (体制多样性的窗口级代理)
标签: 1{hit@3 = 1} (窗口级, 来自 SCM/TE 金标准)
验证: LODO 留一数据集 + LOSO 留一打分器; 模型 LR / HistGB
消融: 全特征 / -全局图特征(防数据集指纹) / -打分器特征 / -一致性特征 / 仅归因特征
输出: _cache/e3_windows.json (样本表), e3_results.json (AUROC/ECE/覆盖曲线)
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.scm import make_sample, build_windows
from data.tep import load_tep_units
from graphs import graph_ops as go
from graphs.sources import pcmci_graph
from graphs.graph_ops import graph_features
from scorers import make_scorer, TorchPCAScorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graph_free import GRAPH_FREE
from evaluation import metrics as M
from scripts.e1_anchor import perturb_graph, perturb_input, stable_seed

CACHE = os.path.join(SRC, '_cache')
M_PERTURB = 6


def phi_shape_feats(phi_row):
    a = np.abs(phi_row) + 1e-12
    p = a / a.sum()
    order = np.argsort(-phi_row)
    ent = -(p * np.log(p)).sum() / np.log(len(p))
    srt = np.sort(p)[::-1]
    return dict(
        top1_share=float(srt[0]),
        top3_share=float(srt[:3].sum()),
        attr_entropy=float(ent),
        top_gap=float(srt[0] - srt[1]),
        top_ratio=float(srt[0] / (srt[1] + 1e-12)),
    )


def scorer_feats(scores_w, scores_pool):
    q = np.percentile(scores_pool, [50, 90, 99])
    return dict(
        score_margin_rel=float((scores_w - q[1]) / (q[1] + 1e-9)),
        score_z=float((scores_w - scores_pool.mean()) / (scores_pool.std() + 1e-9)),
        score_exceed99=float(scores_w > q[2]),
    )


def consistency_feats(phi_rows):
    """phi_rows: (M+1, D) 基准+M扰动。窗口级一致性 (ACR 的窗口级特征)。"""
    base = phi_rows[0]
    overlaps = []
    top1s = []
    for r in phi_rows[1:]:
        overlaps.append(len(M.topk(r, 3) & M.topk(base, 3)) / 3)
        top1s.append(int(np.argmax(r) == np.argmax(base)))
    ranks = np.argsort(-base)
    return dict(
        cons_topk3=float(np.mean(overlaps)),
        cons_top1=float(np.mean(top1s)),
    )


def dev_feats(X_row, ctx):
    """窗口内通道偏差分布 (数据条件代理)。"""
    st = ctx.pool_stats
    z = np.abs(X_row - st['med'][None, :]).mean(0) / st['mad']
    p = z / (z.sum() + 1e-12)
    ent = -(p * np.log(p + 1e-12)).sum() / np.log(len(p))
    srt = np.sort(p)[::-1]
    return dict(dev_entropy=float(ent), dev_top1=float(srt[0]),
                dev_top3=float(srt[:3].sum()))


MAX_WINDOWS = 30          # 每单元采样窗口上限 (控制样本表规模)


def collect_units():
    units = []
    for seed in range(5):
        for mag in (0.25, 0.5):
            s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2, seed=seed,
                            noise_std=0.5, magnitude=mag)
            w = build_windows(s)
            n_w = min(MAX_WINDOWS, len(w['X_anom']))
            sel = np.linspace(0, len(w['X_anom']) - 1, n_w).astype(int)
            units.append(dict(
                name=f'scm{seed}_m{mag}', kind='scm',
                X_anom=w['X_anom'][sel], X_pool=w['X_pool'],
                R_anom=[w['R_anom'][i] for i in sel],
                A_true=s['adjacency'].astype(float),
                series_normal=s['normal'], series_full=s['series']))
    for u0 in load_tep_units():
        n_w = min(MAX_WINDOWS, len(u0['X_anom']))
        sel = np.linspace(0, len(u0['X_anom']) - 1, n_w).astype(int)
        u = dict(u0)
        u['X_anom'] = u0['X_anom'][sel]
        u['R_anom'] = [u0['R_anom'][i] for i in sel]
        cache = os.path.join(CACHE, f"te_graph_{u['name']}.npz")
        A = np.load(cache, allow_pickle=True)['A'] if os.path.exists(cache) \
            else pcmci_graph(u['series_normal'], tau_max=2)['adj']
        u['kind'] = 'te'; u['A_true'] = None
        u['graphs'] = {'pcmci': A}
        units.append(u)
    return units


def build_samples():
    rows = []
    for u in [x for x in collect_units()]:
        graphs = u.get('graphs') or {
            'true': u['A_true'],
            'pcmci_clean': pcmci_graph(u['series_normal'])['adj'],
            'pcmci_anom': pcmci_graph(u['series_full'])['adj'],
        }
        for sk in ('iforest', 'pca', 'ocsvm'):
            scorer = make_scorer(sk).fit(u['X_pool'])
            s_pool = scorer.score(u['X_pool'])
            torch_scorer = TorchPCAScorer(scorer) if sk == 'pca' else None
            ctx0 = Context(u['X_pool'], scorer=scorer)
            ctx0.torch_scorer = torch_scorer
            for mname, fn in list(GRAPH_DEPENDENT.items()) + list(GRAPH_FREE.items()):
                if mname == 'Grad' and torch_scorer is None:
                    continue
                graph_dep = mname in GRAPH_DEPENDENT
                for src_name, A in (graphs.items() if graph_dep else [('none', None)]):
                    ctx_b = ctx0.with_graph(A)
                    phi_b = fn(u['X_anom'], ctx_b)
                    hits = M.hit_at_k(phi_b, u['R_anom'], 3)
                    gfeats = graph_features(A) if graph_dep else {}
                    # 批量一致性: 对每个扰动实例一次性算全部窗口
                    phi_perts = []
                    for mi in range(M_PERTURB):
                        rng = np.random.default_rng(stable_seed(
                            'cons', u['name'], sk, mname, src_name, mi))
                        if graph_dep:
                            fam = ['del', 'add', 'rew'][mi % 3]
                            A_m = perturb_graph(A, fam, 0.2, rng)
                            phi_perts.append(fn(u['X_anom'], ctx0.with_graph(A_m)))
                        else:
                            fam = ['noise', 'slice'][mi % 2]
                            X_m = perturb_input(u['X_anom'], fam,
                                                0.3 if fam == 'noise' else 2,
                                                rng, ctx0)
                            phi_perts.append(fn(X_m, ctx0))
                    for i in range(len(phi_b)):
                        phi_rows = np.stack([phi_b[i]] + [pp[i] for pp in phi_perts])
                        rows.append(dict(
                            dataset=u['name'], data_kind=u['kind'], scorer=sk,
                            method=mname, graph_source=src_name, window=i,
                            label=int(hits[i]),
                            **{f'g_{k}': v for k, v in gfeats.items()},
                            **{f'c_{k}': v for k, v in
                               consistency_feats(np.stack(phi_rows)).items()},
                            **{f'a_{k}': v for k, v in
                               phi_shape_feats(phi_b[i]).items()},
                            **{f's_{k}': v for k, v in scorer_feats(
                                float(scorer.score(u['X_anom'][i:i + 1])[0]), s_pool).items()},
                            **{f'd_{k}': v for k, v in
                               dev_feats(u['X_anom'][i], ctx0).items()},
                        ))
        print(f"  samples so far: {len(rows)} ({u['name']})", flush=True)
    out = os.path.join(CACHE, 'e3_windows.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, default=float)
    print(f"{len(rows)} window samples -> {out}")
    return rows


FEATURE_GROUPS = {
    'graph': lambda c: c.startswith('g_'),
    'consistency': lambda c: c.startswith('c_'),
    'attribution': lambda c: c.startswith('a_'),
    'scorer': lambda c: c.startswith('s_'),
    'data': lambda c: c.startswith('d_'),
}


def fit_predictor(rows, group_drop=(), model='hgb'):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score

    feats = [c for c in rows[0] if c[0] in 'gcasd' and c[1] == '_']
    feats = [f for f in feats
             if not any(pred(f) for g, pred in FEATURE_GROUPS.items() if g in group_drop)]
    X = np.array([[r.get(f, np.nan) for f in feats] for r in rows], dtype=float)
    y = np.array([r['label'] for r in rows])
    datasets = sorted({r['dataset'] for r in rows})
    scorers = sorted({r['scorer'] for r in rows})

    def mk():
        from sklearn.impute import SimpleImputer
        if model == 'lr':
            return make_pipeline(SimpleImputer(strategy='median'),
                                 StandardScaler(),
                                 LogisticRegression(max_iter=2000, C=1.0))
        return HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                              learning_rate=0.08, random_state=0)

    results = []
    # LODO
    for d in datasets:
        tr = np.array([r['dataset'] != d for r in rows])
        te = ~tr
        if y[tr].std() == 0 or y[te].std() == 0:
            continue
        clf = mk().fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        results.append(dict(protocol='LODO', held=d, model=model,
                            drop=list(group_drop),
                            auroc=float(roc_auc_score(y[te], p)),
                            n_test=int(te.sum())))
    # LOSO
    for s in scorers:
        tr = np.array([r['scorer'] != s for r in rows])
        te = ~tr
        if y[tr].std() == 0 or y[te].std() == 0:
            continue
        clf = mk().fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        results.append(dict(protocol='LOSO', held=s, model=model,
                            drop=list(group_drop),
                            auroc=float(roc_auc_score(y[te], p)),
                            n_test=int(te.sum())))
    return results, feats


def coverage_curve(rows, model='hgb', group_drop=()):
    """选择性覆盖: LODO 预测概率作可信度, 覆盖率-正确率曲线。"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    import collections

    by_ds = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_ds[r['dataset']].append(i)
    feats = [c for c in rows[0] if c[0] in 'gcasd' and c[1] == '_']
    feats = [f for f in feats
             if not any(pred(f) for g, pred in FEATURE_GROUPS.items() if g in group_drop)]
    X = np.array([[r.get(f, np.nan) for f in feats] for r in rows], dtype=float)
    y = np.array([r['label'] for r in rows])
    prob = np.full(len(rows), np.nan)
    for d, idx in by_ds.items():
        te = np.zeros(len(rows), dtype=bool); te[idx] = True
        tr = ~te
        if y[tr].std() == 0 or y[te].std() == 0:
            continue
        if model == 'lr':
            from sklearn.impute import SimpleImputer
            clf = make_pipeline(SimpleImputer(strategy='median'),
                                StandardScaler(),
                                LogisticRegression(max_iter=2000))
        else:
            clf = HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                                 learning_rate=0.08, random_state=0)
        clf.fit(X[tr], y[tr])
        prob[te] = clf.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(prob)
    order = np.argsort(-prob[ok])
    ys = y[ok][order]
    ps = prob[ok][order]
    curve = []
    for cov in np.arange(0.1, 1.01, 0.1):
        k = max(1, int(round(cov * len(ys))))
        curve.append(dict(coverage=float(k / len(ys)),
                          accuracy=float(ys[:k].mean()),
                          threshold=float(ps[k - 1])))
    return curve


def main():
    cache = os.path.join(CACHE, 'e3_windows.json')
    if '--reuse' in sys.argv and os.path.exists(cache):
        with open(cache, encoding='utf-8') as f:
            rows = json.load(f)
        print(f"loaded {len(rows)} cached samples")
    else:
        rows = build_samples()
    out = {'results': [], 'coverage': [], 'importance': None}
    for model in ('hgb', 'lr'):
        for drop in ((), ('graph',), ('scorer',), ('consistency',),
                     ('graph', 'scorer'), ('consistency', 'scorer')):
            res, feats = fit_predictor(rows, group_drop=drop, model=model)
            import collections
            agg = collections.defaultdict(list)
            for r in res:
                agg[r['protocol']].append(r['auroc'])
            summary = {k: float(np.mean(v)) for k, v in agg.items()}
            out['results'].append(dict(model=model, drop=list(drop),
                                       n_feats=len(feats), summary=summary,
                                       detail=res))
            print(model, drop, summary, flush=True)
    out['coverage'] = coverage_curve(rows)
    with open(os.path.join(CACHE, 'e3_results.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print('-> e3_results.json')


if __name__ == '__main__':
    main()
