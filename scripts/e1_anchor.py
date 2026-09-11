# -*- coding: utf-8 -*-
"""E1 锚点实验: 合成 SCM × 图源 × 扰动 → hit@K / ACR@K 双指标, 验证解耦象限。

开工前置锚点 (提示词): 确认 "高 ACR(>0.8) 低 hit@3(<0.4)" 解耦象限存在。
通过 → 全量开工; 不通过 → 主轴降级 (退化曲线+膝盖+预测器)。

设计:
  数据     SCM n=15, T=5000, seeds 0..4, 每轨迹 5 单根因 + 2 双根因
  图源     true / pcmci_clean / pcmci_anom (系统性偏差源)
  打分器   iforest / pca (Grad 仅 pca-torch)
  图依赖   DPTA-G, PropRank, GraphGranger  → 图扰动 (del/add/rew × 强度 × M=8)
  图无关   GlobalCF, AERec, Grad, zDev, Random → 输入扰动 (noise/slice × M=8)
  指标     hit@1/3/5 (phi(G_base)); ACR@3 双参照:
             ACR_base = <overlap(phi(G_m~P(G_base)), phi(G_base))>   部署场景
             ACR_true = <overlap(phi(G_m~P(G_true)),  phi(G_true))>  学术场景
输出: src/_cache/e1_anchor.json (每 cell 一行: 全部指标 + 象限标记)
"""
import os
import sys
import json
import time
import zlib

import numpy as np


def stable_seed(*parts) -> int:
    return zlib.crc32('|'.join(str(p) for p in parts).encode())

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)          # .../src
sys.path.insert(0, SRC)

from data.scm import make_sample, build_windows
from graphs.sources import graph_sources
from graphs import graph_ops as go
from scorers import make_scorer, TorchPCAScorer, detection_auroc
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graph_free import GRAPH_FREE
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')
os.makedirs(CACHE, exist_ok=True)

N_SEEDS = 5
GRAPH_STRENGTHS = {'del': [0.1, 0.2, 0.3, 0.5], 'add': [0.2, 0.5], 'rew': [0.2, 0.5]}
INPUT_STRENGTHS = dict(noise=[0.2, 0.5], slice=[1, 3])
M_PERTURB = 8
KS = [1, 3, 5]


def perturb_graph(A, family, strength, rng):
    if family == 'del':
        return go.delete_edges(A, strength, rng)
    if family == 'add':
        return go.add_edges(A, strength, rng)
    if family == 'rew':
        return go.rewire_edges(A, strength, rng)
    raise ValueError(family)


def perturb_input(X, family, strength, rng, ctx):
    if family == 'noise':
        mad = ctx.pool_stats['mad']
        return (X + rng.normal(0, 1, X.shape).astype(np.float32)
                * (strength * mad)[None, None, :]).astype(np.float32)
    if family == 'slice':
        # 时间切片: 循环平移窗口内通道时序 (破坏时间对齐, 保留边际)
        out = np.empty_like(X)
        n, W, D = X.shape
        for i in range(n):
            for j in range(D):
                s = rng.integers(-strength, strength + 1)
                out[i, :, j] = np.roll(X[i, :, j], int(s))
        return out
    raise ValueError(family)


def cell_record(dataset, scorer_kind, method, family, strength, source,
                phi_hit, roots, acrs_base, acrs_true=None):
    rec = dict(
        dataset=dataset, scorer=scorer_kind, method=method,
        family=family, strength=float(strength), graph_source=source,
        n_windows=len(roots),
        hit1=float(M.mean_hit(phi_hit, roots, 1)),
        hit3=float(M.mean_hit(phi_hit, roots, 3)),
        hit5=float(M.mean_hit(phi_hit, roots, 5)),
        rbo=float(M.mean_rbo(phi_hit, roots)),
        acr3_base=float(np.mean(acrs_base)) if len(acrs_base) else None,
    )
    if acrs_true is not None and len(acrs_true):
        rec['acr3_true'] = float(np.mean(acrs_true))
    return rec


MAGS = [0.25, 0.5, 1.5]          # 弱/中/强注入 (noise_std=0.5) —— 数据条件分层


def run_seed(seed, mag=0.5, verbose=True):
    t0 = time.time()
    sample = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2, seed=seed,
                         noise_std=0.5, magnitude=mag)
    win = build_windows(sample, window=16, stride=20)
    X_anom, X_pool, roots = win['X_anom'], win['X_pool'], win['R_anom']
    if verbose:
        print(f"[seed {seed}] anomalous windows={len(X_anom)} pool={len(X_pool)}",
              flush=True)
    sources = graph_sources(sample)

    records = []
    for sk in ('iforest', 'pca'):
        scorer = make_scorer(sk).fit(X_pool)
        auroc = detection_auroc(scorer, X_pool, X_anom)
        torch_scorer = None
        if sk == 'pca':
            torch_scorer = TorchPCAScorer(scorer)
        ctx0 = Context(X_pool, scorer=scorer)
        ctx0.torch_scorer = torch_scorer
        rng_master = np.random.default_rng(seed * 100 + hash(sk) % 97)

        # ---------- 图依赖方法: 图扰动 ----------
        for mname, fn in GRAPH_DEPENDENT.items():
            # 场景: base = true (学术) 与 base = pcmci_* (部署)
            for src_name, A_base in sources.items():
                ctx_base = ctx0.with_graph(A_base)
                phi_base = fn(X_anom, ctx_base)
                # hit 用 phi(G_base) (部署图)
                rec_hit = dict(
                    dataset=f'scm{seed}_m{mag}', scorer=sk, method=mname, family='none',
                    strength=0.0, graph_source=src_name, n_windows=len(roots),
                    hit1=float(M.mean_hit(phi_base, roots, 1)),
                    hit3=float(M.mean_hit(phi_base, roots, 3)),
                    hit5=float(M.mean_hit(phi_base, roots, 5)),
                    rbo=float(M.mean_rbo(phi_base, roots)),
                    acr3_base=None, acr3_true=None)
                records.append(rec_hit)
                # 扰动族
                for fam, strengths in GRAPH_STRENGTHS.items():
                    for st in strengths:
                        acrs = []
                        for mi in range(M_PERTURB):
                            rng = np.random.default_rng(
                                stable_seed(seed, sk, mname, src_name, fam, st, mi))
                            A_m = perturb_graph(A_base, fam, st, rng)
                            ctx_m = ctx0.with_graph(A_m)
                            phi_m = fn(X_anom, ctx_m)
                            acrs.append(M.acr_at_k(phi_m, phi_base, 3).mean())
                        hit3p = []
                        for mi2 in range(M_PERTURB):
                            rng2 = np.random.default_rng(stable_seed(
                                seed, sk, mname, src_name, fam, st, mi2))
                            A_m2 = perturb_graph(A_base, fam, st, rng2)
                            phi_m2 = fn(X_anom, ctx0.with_graph(A_m2))
                            hit3p.append(float(M.mean_hit(phi_m2, roots, 3)))
                        rr = cell_record(
                            f'scm{seed}_m{mag}', sk, mname, fam, st, src_name,
                            phi_base, roots, acrs)
                        rr['hit3_pert'] = float(np.mean(hit3p))
                        records.append(rr)
                if verbose:
                    h3 = rec_hit['hit3']
                    print(f"  [{sk}] {mname:14s} src={src_name:12s} hit@3={h3:.2f}",
                          flush=True)

        # ---------- ACR_true (仅 base=true 场景额外记录相对真值) ----------
        # 已在上面 base=true 的格中: acr3_base 即相对 true —— 命名双参照在分析阶段拆

        # ---------- 图无关方法: 输入扰动 ----------
        for mname, fn in GRAPH_FREE.items():
            if mname == 'Grad' and torch_scorer is None:
                continue
            phi_base = fn(X_anom, ctx0)
            rec_hit = dict(
                dataset=f'scm{seed}_m{mag}', scorer=sk, method=mname, family='none',
                strength=0.0, graph_source='none', n_windows=len(roots),
                hit1=float(M.mean_hit(phi_base, roots, 1)),
                hit3=float(M.mean_hit(phi_base, roots, 3)),
                hit5=float(M.mean_hit(phi_base, roots, 5)),
                rbo=float(M.mean_rbo(phi_base, roots)),
                acr3_base=None, acr3_true=None)
            records.append(rec_hit)
            for fam, strengths in INPUT_STRENGTHS.items():
                for st in strengths:
                    acrs = []
                    for mi in range(M_PERTURB):
                        rng = np.random.default_rng(
                            stable_seed(seed, sk, mname, fam, st, mi))
                        X_m = perturb_input(X_anom, fam, st, rng, ctx0)
                        phi_m = fn(X_m, ctx0)
                        acrs.append(M.acr_at_k(phi_m, phi_base, 3).mean())
                    hit3p = []
                    for mi2 in range(M_PERTURB):
                        rng2 = np.random.default_rng(stable_seed(
                            seed, sk, mname, fam, st, mi2))
                        X_m2 = perturb_input(X_anom, fam, st, rng2, ctx0)
                        phi_m2 = fn(X_m2, ctx0)
                        hit3p.append(float(M.mean_hit(phi_m2, roots, 3)))
                    rr = cell_record(
                        f'scm{seed}_m{mag}', sk, mname, fam, st, 'none',
                        phi_base, roots, acrs)
                    rr['hit3_pert'] = float(np.mean(hit3p))
                    records.append(rr)
            if verbose:
                print(f"  [{sk}] {mname:14s} input-pert hit@3={rec_hit['hit3']:.2f}",
                      flush=True)
        records.append(dict(dataset=f'scm{seed}_m{mag}', scorer=sk, method='_meta',
                            det_auroc=auroc))
    print(f"[seed {seed}] done in {time.time()-t0:.0f}s, cells={len(records)}",
          flush=True)
    return records


def main():
    all_records = []
    for mag in MAGS:
        for seed in range(N_SEEDS):
            all_records.extend(run_seed(seed, mag=mag))
    out = os.path.join(CACHE, 'e1_anchor.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(all_records, f, ensure_ascii=False, indent=1, default=float)
    print(f"\nTotal records: {len(all_records)} -> {out}")

    # ---------- 快速象限分析 ----------
    import collections
    cells = collections.defaultdict(list)
    for r in all_records:
        if r.get('method', '').startswith('_') or r['family'] == 'none':
            continue
        if r['acr3_base'] is None:
            continue
        cells[(r['dataset'], r['scorer'], r['method'], r['family'],
               r['strength'], r['graph_source'])].append(
            (r['acr3_base'], r['hit3'], r['method']))
    pts = [(np.mean([a for a, h, _ in v]), np.mean([h for a, h, _ in v]))
           for v in cells.values()]
    q = M.quadrant_stats(pts)
    print("\n=== Quadrant (cell-level, ACR_base vs hit@3 of base graph) ===")
    print(json.dumps(q, indent=1))
    # 分图依赖/图无关
    for grp, names in (('graph-dep', set(GRAPH_DEPENDENT)),
                       ('graph-free', set(GRAPH_FREE))):
        sub = [[(a, h) for a, h, m in v if m in names]
               for v in cells.values()]
        sub = [x for x in sub if x]
        pts_g = [(np.mean([a for a, _ in x]), np.mean([h for _, h in x]))
                 for x in sub]
        print(grp, json.dumps(M.quadrant_stats(pts_g)))


if __name__ == '__main__':
    main()
