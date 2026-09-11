# -*- coding: utf-8 -*-
"""E65 (修订弹药 A): AttNN 注意力代理深度基线 — SCM medium 双指标评测.

协议对齐 e1_anchor 的图无关族分支:
  数据   SCM n=15 T=5000 noise_std=0.5 magnitude=0.5 (medium), seeds 0..4
  方法   AttNN (主) + Random 对照
  打分器 iforest / pca
  hit@3  phi(X_anom)
  ACR@3  输入扰动 (noise 0.2/0.5×MAD, slice 1/3 循环移位; 各 M=8) 下
         attention(φ(X~P)) 与 φ(X) 的 top-3 重叠; 代理模型不随扰动重训练
         (与图依赖族"同一打分器换扰动图"的公平性口径一致)
φ 变体: alpha (主) / alpha_mad (备); 选用规则见 EXECUTION_PLAN 2.1 —
  若 (seed0, iforest) 上 alpha 的 hit@3 < 0.15, 取该 cell 上两者 hit@3 高者,
  全局统一应用 (不做逐 cell 挑选)。
图源: 图无关方法与图无关 (记 graph='none'), 不重复 true/pcmci (协议修正,
  见 PROGRESS.md 2026-09-07 条目)。
输出: _cache/e65_attnn.json
用法:  python scripts/e65_attnn.py            # 全量 (5 seeds × 2 scorers)
       E65_SMOKE=1 python scripts/e65_attnn.py out2.json   # 冒烟/第二输出名
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
import sys
import json
import time
import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.scm import make_sample, build_windows
from scorers import make_scorer
from attribution import Context
from attribution.graph_free import random_attribute
from attribution.attnn import attnn_alpha, MASTER_SEED
from scripts.e1_anchor import stable_seed, perturb_input
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')
INPUT_STRENGTHS = dict(noise=[0.2, 0.5], slice=[1, 3])
M_PERT = 8


def run(out_name='e65_attnn.json'):
    smoke = os.environ.get('E65_SMOKE') == '1'
    seeds = [0] if smoke else list(range(5))
    scorers = ('iforest',) if smoke else ('iforest', 'pca')
    t_all = time.time()
    cells, rand = [], []
    variant = None
    for seed in seeds:
        sample = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                             seed=seed, noise_std=0.5, magnitude=0.5)
        win = build_windows(sample, window=16, stride=20)
        X_anom, X_pool, roots = win['X_anom'], win['X_pool'], win['R_anom']
        for sk in scorers:
            t0 = time.time()
            scorer = make_scorer(sk).fit(X_pool)
            ctx0 = Context(X_pool, scorer=scorer)
            alpha_fn, info = attnn_alpha(X_anom, ctx0, seed=MASTER_SEED)
            mad = ctx0.pool_stats['mad']
            phi_a = alpha_fn(X_anom)
            phi_m = phi_a * mad[None, :]
            hit_a = float(M.mean_hit(phi_a, roots, 3))
            hit_m = float(M.mean_hit(phi_m, roots, 3))
            if variant is None:                      # 首个 cell 定全局变体
                variant = 'alpha' if hit_a >= 0.15 else (
                    'alpha' if hit_a >= hit_m else 'alpha_mad')
                print(f'[variant] hit_alpha={hit_a:.3f} hit_mad={hit_m:.3f} '
                      f'-> {variant}', flush=True)
            sel = phi_a if variant == 'alpha' else phi_m
            sel_hit = hit_a if variant == 'alpha' else hit_m
            acr_by = {}
            for fam, sts in INPUT_STRENGTHS.items():
                for st in sts:
                    acrs = []
                    for mi in range(M_PERT):
                        rng = np.random.default_rng(stable_seed(
                            'e65', seed, sk, fam, st, mi))
                        Xm = perturb_input(X_anom, fam, st, rng, ctx0)
                        phim = alpha_fn(Xm)
                        if variant == 'alpha_mad':
                            phim = phim * mad[None, :]
                        acrs.append(float(M.acr_at_k(phim, sel, 3).mean()))
                    acr_by[f'{fam}@{st}'] = float(np.mean(acrs))
            cells.append(dict(
                seed=seed, graph='none', scorer=sk, method='attnn',
                variant=variant, hit3=sel_hit, hit3_alpha=hit_a,
                hit3_alpha_mad=hit_m, acr3_input=float(np.mean(
                    list(acr_by.values()))), acr_breakdown=acr_by,
                train=info, wall_s=round(time.time() - t0, 1)))
            # ---- Random 对照 (同扰动协议) ----
            t1 = time.time()
            rphi = random_attribute(X_anom, ctx0, seed=int(
                stable_seed('e65', seed, sk, 'rand')))
            racr = []
            for fam, sts in INPUT_STRENGTHS.items():
                for st in sts:
                    acrs = []
                    for mi in range(M_PERT):
                        rng = np.random.default_rng(stable_seed(
                            'e65', seed, sk, 'rand', fam, st, mi))
                        Xm = perturb_input(X_anom, fam, st, rng, ctx0)
                        rm = random_attribute(Xm, ctx0, seed=int(
                            stable_seed('e65', seed, sk, 'rand', fam, st, mi)))
                        acrs.append(float(M.acr_at_k(rm, rphi, 3).mean()))
                    racr.append(np.mean(acrs))
            rand.append(dict(seed=seed, graph='none', scorer=sk,
                             method='Random', hit3=float(M.mean_hit(
                                 rphi, roots, 3)),
                             acr3_input=float(np.mean(racr)),
                             wall_s=round(time.time() - t1, 1)))
            print(f'[seed {seed} {sk}] attnn hit@3={sel_hit:.3f} '
                  f'acr3_input={cells[-1]["acr3_input"]:.3f} '
                  f'({cells[-1]["wall_s"]}s) | rand hit@3={rand[-1]["hit3"]:.3f}',
                  flush=True)
    out = dict(
        meta=dict(script='e65_attnn',
                  date=datetime.datetime.now().isoformat(timespec='seconds'),
                  variant=variant, smoke=smoke,
                  torch=__import__('torch').__version__,
                  wall_total_s=round(time.time() - t_all, 1)),
        cells=cells, random_control=rand)
    path = os.path.join(CACHE, out_name)
    json.dump(out, open(path, 'w'), default=float, ensure_ascii=False, indent=1)
    print(f'-> {path}  cells={len(cells)} rand={len(rand)} '
          f'total={out["meta"]["wall_total_s"]}s')
    # ---- 汇总 ----
    import collections
    agg = collections.defaultdict(list)
    for c in cells:
        agg[(c['method'], c['variant'], c['scorer'])].append(
            (c['hit3'], c['acr3_input']))
    for r in rand:
        agg[('Random', '-', r['scorer'])].append((r['hit3'], r['acr3_input']))
    print('\n=== E65 summary (SCM medium, input-pert ACR@3) ===')
    for k in sorted(agg, key=str):
        v = agg[k]
        print(f'  {str(k):34} hit@3={np.mean([x[0] for x in v]):.3f} '
              f'acr3={np.mean([x[1] for x in v]):.3f} (n={len(v)})')


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else 'e65_attnn.json'
    run(name)
