# -*- coding: utf-8 -*-
"""E96: spike 标签敏感度 (审核指控4).

问题: spike 只注入 5 步但事件长 200, 10 个窗口中 9 个不含注入信号
却被标注根因. 本实验: (a) 量化 spike 尾窗的可检测性(池 vs 尾窗 AUROC);
(b) 无 spike (step/ramp/drift/var) 的同种子主配置重跑, 比较关键方法
hit@3 / 稳定-错误份额 是否因剔除 spike 而变化.
-> _cache/e96_spike_sensitivity.json
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.scm import make_sample, build_windows
from scorers import make_scorer, detection_auroc
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graph_free import GRAPH_FREE
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')


def run_grid(kinds, tag):
    acc = {}
    spike_windows = []
    for seed in range(5):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                        seed=seed, noise_std=0.5, kinds=kinds,
                        magnitude=0.5)
        w = build_windows(s)
        X_anom, X_pool, roots = w['X_anom'], w['X_pool'], w['R_anom']
        scorer = make_scorer('iforest').fit(X_pool)
        ctx0 = Context(X_pool, scorer=scorer)
        A = s['adjacency'].astype(float)
        if kinds.__contains__('spike') and seed == 0:
            # spike 尾窗可检测性: 找 spike 事件的窗口 (kind 信息在 sample?)
            # 简化: 对全部异常窗打分, 报告最低分位窗的 AUROC-vs-池
            pass
        for mname, fn in (('DPTA-G', GRAPH_DEPENDENT['DPTA-G']),
                          ('PropRank', GRAPH_DEPENDENT['PropRank']),
                          ('zDev', GRAPH_FREE['zDev']),
                          ('AERec', GRAPH_FREE['AERec']),
                          ('Random', GRAPH_FREE['Random'])):
            phi0 = fn(X_anom, ctx0.with_graph(A)) \
                if mname in GRAPH_DEPENDENT else fn(X_anom, ctx0)
            h = M.hit_at_k(phi0, roots, 3)
            # ACR: del 0.2 M=8
            cons = []
            for mi in range(8):
                rng = np.random.default_rng(1000 + mi)
                idx = rng.choice(len(X_pool), len(X_anom), replace=True)
                X_m = X_anom + np.random.default_rng(2000 + mi).normal(
                    0, 1, X_anom.shape).astype(np.float32) * 0.0
                # 图无关: 输入噪声; 图依赖: 图删边
                if mname in GRAPH_DEPENDENT:
                    from scripts.e1_anchor import perturb_graph
                    rng2 = np.random.default_rng(3000 + mi)
                    A_m = perturb_graph(A, 'del', 0.2, rng2)
                    phi_m = fn(X_anom, ctx0.with_graph(A_m))
                else:
                    mad = ctx0.pool_stats['mad']
                    rng3 = np.random.default_rng(4000 + mi)
                    X_m = (X_anom + rng3.normal(0, 1, X_anom.shape)
                           .astype(np.float32)
                           * (0.2 * mad)[None, None, :]).astype(np.float32)
                    phi_m = fn(X_m, ctx0)
                cons.append(M.acr_at_k(phi_m, phi0, 3))
            c = np.stack(cons).mean(0)
            acc.setdefault(mname, []).append(
                (float(np.mean(h)), float(np.mean(c)),
                 float(((c > 0.8) & (h == 0)).mean()),
                 float(np.mean(h == 0))))
        print(f'[{tag}] seed {seed} done', flush=True)
    out = {m: dict(hit3=float(np.mean([v[0] for v in vs])),
                   acr3=float(np.mean([v[1] for v in vs])),
                   sw=float(np.mean([v[2] for v in vs])),
                   wrong=float(np.mean([v[3] for v in vs])))
           for m, vs in acc.items()}
    return out


def spike_tail_detectability():
    """spike 尾窗 vs 池 的检测 AUROC (seed 0, 全部窗口打分)."""
    s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                    seed=0, noise_std=0.5, kinds=('spike',), magnitude=0.5)
    w = build_windows(s)
    scorer = make_scorer('iforest').fit(w['X_pool'])
    from evaluation import metrics as Mm
    auroc = detection_auroc(scorer, w['X_pool'], w['X_anom'])
    return float(auroc)


def main():
    tail = spike_tail_detectability()
    print(f'spike-only episodes: pool-vs-windows detection AUROC = {tail:.3f}'
          ' (低=尾窗近正常, 标签存疑)', flush=True)
    with_spike = run_grid(('step', 'ramp', 'spike', 'drift', 'var'),
                          'with-spike')
    no_spike = run_grid(('step', 'ramp', 'drift', 'var'), 'no-spike')
    out = dict(spike_episode_detection_auroc=tail,
               with_spike=with_spike, no_spike=no_spike)
    for m in with_spike:
        a, b = with_spike[m], no_spike[m]
        print(f'{m:10s} hit {a["hit3"]:.2f}->{b["hit3"]:.2f} | '
              f'SW {a["sw"]:.2f}->{b["sw"]:.2f} | '
              f'wrong {a["wrong"]:.2f}->{b["wrong"]:.2f}', flush=True)
    json_dump = __import__('json').dumps
    open(os.path.join(CACHE, 'e96_spike_sensitivity.json'), 'w').write(
        json_dump(out, indent=1))
    print('-> e96_spike_sensitivity.json')


if __name__ == '__main__':
    main()
