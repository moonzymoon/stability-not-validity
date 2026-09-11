# -*- coding: utf-8 -*-
"""E81 案例研究: 单窗口双方法归因, 供补充材料 Figure S1。

从释放版管线(seed 0, medium, iforest)取一个窗口, 满足:
  AERec: 稳定-错误 (跨6次噪声扰动 top-3 重叠高, 但 top-3 不含真根因)
  zDev:  正确 (top-3 含真根因)
输出完整逐变量分数 + 一致性值 -> _cache/e81_case_study.json
图由 make_fig_case.py 从本缓存生成 (不变量: 每图来自缓存 JSON)。
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.scm import make_sample, build_windows
from scorers import make_scorer
from attribution import Context
from attribution.graph_free import GRAPH_FREE
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')


def top3(scores):
    return list(np.argsort(-scores)[:3])


def main():
    sample = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                         seed=0, noise_std=0.5, magnitude=0.5)
    win = build_windows(sample, window=16, stride=20)
    X_anom, X_pool, roots = win['X_anom'], win['X_pool'], win['R_anom']
    scorer = make_scorer('iforest').fit(X_pool)
    ctx = Context(X_pool, scorer=scorer)

    zdev = GRAPH_FREE['zDev'](X_anom, ctx)
    aerec = GRAPH_FREE['AERec'](X_anom, ctx)

    # AERec 一致性: 6 次输入噪声扰动 (strength 0.2, 同 e1 声明带)
    rng_master = np.random.default_rng(20260909)
    mad = ctx.pool_stats['mad']
    overlaps = []
    for mi in range(6):
        rng = np.random.default_rng(
            int(rng_master.integers(0, 2**31 - 1)))
        X_m = (X_anom + rng.normal(0, 1, X_anom.shape).astype(np.float32)
               * (0.2 * mad)[None, None, :]).astype(np.float32)
        phi_m = GRAPH_FREE['AERec'](X_m, ctx)
        overlaps.append(float(M.acr_at_k(phi_m, aerec, 3).mean()))

    # 找一个窗口: AERec 稳定-错误, zDev 正确
    def rootset(w):
        return set(int(x) for x in roots[w])

    def window_draws(w, n=6):
        out = []
        rng2 = np.random.default_rng(20260909)
        for _ in range(n):
            rng = np.random.default_rng(int(rng2.integers(0, 2**31 - 1)))
            X_m = (X_anom + rng.normal(0, 1, X_anom.shape).astype(np.float32)
                   * (0.2 * mad)[None, None, :]).astype(np.float32)
            phi_m = GRAPH_FREE['AERec'](X_m, ctx)
            t0, t1 = top3(aerec[w]), top3(phi_m[w])
            out.append(len(set(t0) & set(t1)) / 3.0)
        return out

    pick = None
    for w in range(len(roots)):
        r = rootset(w)
        a3, z3 = set(top3(aerec[w])), set(top3(zdev[w]))
        if (a3 & r) or not (z3 & r):
            continue
        draws = window_draws(w)
        if np.mean(draws) >= 0.67:
            pick = (w, r, a3, z3, draws)
            break

    if pick is None:
        for w in range(len(roots)):
            r = rootset(w)
            a3, z3 = set(top3(aerec[w])), set(top3(zdev[w]))
            if not (a3 & r) and (z3 & r):
                pick = (w, r, a3, z3, window_draws(w))
                break

    w, r, a3, z3, draws = pick
    rec = dict(
        seed=0, magnitude=0.5, scorer='iforest', window_index=int(w),
        true_roots=[int(x) for x in sorted(r)],
        aerec_top3=[int(x) for x in sorted(a3)],
        zdev_top3=[int(x) for x in sorted(z3)],
        aerec_scores=[float(x) for x in aerec[w]],
        zdev_scores=[float(x) for x in zdev[w]],
        aerec_window_consistency=float(np.mean(draws)),
        aerec_batch_acr3=float(np.mean(overlaps)),
    )
    out = os.path.join(CACHE, 'e81_case_study.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(rec, f, indent=1)
    print('window', w, '| roots', sorted(r), '| AERec top3', sorted(a3),
          '| zDev top3', sorted(z3),
          '| AERec consistency %.2f' % rec['aerec_window_consistency'],
          '| batch ACR@3 %.2f' % rec['aerec_batch_acr3'])
    print('->', out)


if __name__ == '__main__':
    main()
