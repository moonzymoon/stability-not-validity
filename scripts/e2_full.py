# -*- coding: utf-8 -*-
"""E2 全量实验: (a) TE 真实数据 6 单元; (b) SCM + TE 打分器质量分层。

打分器质量分层 (v2 必改: 隔离"窗口选错"与"归因错"):
  固定金标准异常窗口集合, 仅换打分器 (3 族 × train_frac 3 档 = 9 配置);
  质量操作化 = 检测 AUROC (正常池 vs 异常窗), 作为连续协变量进入分析。
TE 图源: PCMCI(正常训练段) 学图 —— 无真值图, 纯部署场景 (ACR_base only)。
输出: src/_cache/e2_te.json, e2_scm_quality.json
"""
import os
import sys
import json
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.tep import load_tep_units
from data.scm import make_sample, build_windows
from graphs import graph_ops as go
from graphs.sources import pcmci_graph
from scorers import make_scorer, TorchPCAScorer, detection_auroc
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graph_free import GRAPH_FREE
from evaluation import metrics as M
from scripts.e1_anchor import (perturb_graph, perturb_input, cell_record,
                               stable_seed, GRAPH_STRENGTHS, INPUT_STRENGTHS,
                               M_PERTURB)

CACHE = os.path.join(SRC, '_cache')


def run_units(units, tag, graph_of_unit, out_name, scorers=('iforest', 'pca', 'ocsvm'),
              fracs=(0.25, 0.5, 1.0), verbose=True):
    """units: list of dict(name, X_anom, X_pool, R_anom).
    graph_of_unit(unit) -> dict(name->adj)  该单元可用的图源。"""
    records = []
    for u in units:
        t0 = time.time()
        graphs = graph_of_unit(u)
        if verbose:
            print(f"\n=== [{tag}] {u['name']} graphs="
                  f"{ {k: int(a.sum()) for k, a in graphs.items()} }", flush=True)
        for sk in scorers:
            for frac in fracs:
                scorer = make_scorer(sk, train_frac=frac).fit(u['X_pool'])
                auroc = detection_auroc(scorer, u['X_pool'], u['X_anom'])
                torch_scorer = TorchPCAScorer(scorer) if sk == 'pca' else None
                ctx0 = Context(u['X_pool'], scorer=scorer)
                ctx0.torch_scorer = torch_scorer
                for mname, fn in GRAPH_DEPENDENT.items():
                    for src_name, A_base in graphs.items():
                        ctx_b = ctx0.with_graph(A_base)
                        phi_b = fn(u['X_anom'], ctx_b)
                        records.append(dict(
                            dataset=u['name'], tag=tag, scorer=sk, train_frac=frac,
                            method=mname, family='none', strength=0.0,
                            graph_source=src_name, n_windows=len(u['R_anom']),
                            hit1=float(M.mean_hit(phi_b, u['R_anom'], 1)),
                            hit3=float(M.mean_hit(phi_b, u['R_anom'], 3)),
                            hit5=float(M.mean_hit(phi_b, u['R_anom'], 5)),
                            rbo=float(M.mean_rbo(phi_b, u['R_anom'])),
                            det_auroc=auroc))
                        for fam, strengths in GRAPH_STRENGTHS.items():
                            for st in strengths:
                                acrs = []
                                for mi in range(M_PERTURB):
                                    rng = np.random.default_rng(stable_seed(
                                        u['name'], sk, frac, mname, src_name, fam, st, mi))
                                    A_m = perturb_graph(A_base, fam, st, rng)
                                    phi_m = fn(u['X_anom'], ctx0.with_graph(A_m))
                                    acrs.append(M.acr_at_k(phi_m, phi_b, 3).mean())
                                hp = []
                                for mi2 in range(M_PERTURB):
                                    rng2 = np.random.default_rng(stable_seed(
                                        u['name'], sk, frac, mname, src_name, fam, st, mi2))
                                    A_m2 = perturb_graph(A_base, fam, st, rng2)
                                    phi_m2 = fn(u['X_anom'], ctx0.with_graph(A_m2))
                                    hp.append(float(M.mean_hit(phi_m2, u['R_anom'], 3)))
                                records.append(cell_record(
                                    u['name'], sk, mname, fam, st, src_name,
                                    phi_b, u['R_anom'], acrs) | dict(
                                    tag=tag, train_frac=frac, det_auroc=auroc,
                                    hit3_pert=float(np.mean(hp))))
                for mname, fn in GRAPH_FREE.items():
                    if mname == 'Grad' and torch_scorer is None:
                        continue
                    phi_b = fn(u['X_anom'], ctx0)
                    records.append(dict(
                        dataset=u['name'], tag=tag, scorer=sk, train_frac=frac,
                        method=mname, family='none', strength=0.0,
                        graph_source='none', n_windows=len(u['R_anom']),
                        hit1=float(M.mean_hit(phi_b, u['R_anom'], 1)),
                        hit3=float(M.mean_hit(phi_b, u['R_anom'], 3)),
                        hit5=float(M.mean_hit(phi_b, u['R_anom'], 5)),
                        rbo=float(M.mean_rbo(phi_b, u['R_anom'])),
                        det_auroc=auroc))
                    for fam, strengths in INPUT_STRENGTHS.items():
                        for st in strengths:
                            acrs = []
                            for mi in range(M_PERTURB):
                                rng = np.random.default_rng(stable_seed(
                                    u['name'], sk, frac, mname, fam, st, mi))
                                X_m = perturb_input(u['X_anom'], fam, st, rng, ctx0)
                                phi_m = fn(X_m, ctx0)
                                acrs.append(M.acr_at_k(phi_m, phi_b, 3).mean())
                            hp = []
                            for mi2 in range(M_PERTURB):
                                rng2 = np.random.default_rng(stable_seed(
                                    u['name'], sk, frac, mname, fam, st, mi2))
                                X_m2 = perturb_input(u['X_anom'], fam, st, rng2, ctx0)
                                phi_m2 = fn(X_m2, ctx0)
                                hp.append(float(M.mean_hit(phi_m2, u['R_anom'], 3)))
                            records.append(cell_record(
                                u['name'], sk, mname, fam, st, 'none',
                                phi_b, u['R_anom'], acrs) | dict(
                                tag=tag, train_frac=frac, det_auroc=auroc,
                                hit3_pert=float(np.mean(hp))))
        if verbose:
            print(f"  {u['name']} done in {time.time()-t0:.0f}s, total={len(records)}",
                  flush=True)
    out = os.path.join(CACHE, out_name)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=1, default=float)
    print(f"[{tag}] {len(records)} records -> {out}", flush=True)
    return records


def te_graphs(u):
    """TE: PCMCI 在正常训练段学图 (tau_max=2 控制计算量)。缓存。"""
    cache = os.path.join(CACHE, f"te_graph_{u['name']}.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return {'pcmci': d['A']}
    r = pcmci_graph(u['series_normal'], tau_max=2)
    np.savez(cache, A=r['adj'])
    return {'pcmci': r['adj']}


def main():
    # ---- E2a: TE ----
    units = load_tep_units()
    run_units(units, tag='te', graph_of_unit=te_graphs, out_name='e2_te.json')

    # ---- E2b: SCM 打分器质量分层 (中档 mag=0.5) ----
    scm_units = []
    for seed in range(5):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2, seed=seed,
                        noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        A_true = s['adjacency'].astype(float)
        scm_units.append(dict(
            name=f'scm{seed}', X_anom=w['X_anom'], X_pool=w['X_pool'],
            R_anom=w['R_anom'], A_true=A_true,
            series_normal=s['normal'], series_full=s['series']))
    run_units(scm_units, tag='scm',
              graph_of_unit=lambda u: {'true': u['A_true'],
                                       'pcmci_clean': pcmci_graph(u['series_normal'])['adj'],
                                       'pcmci_anom': pcmci_graph(u['series_full'])['adj']},
              out_name='e2_scm_quality.json')


if __name__ == '__main__':
    main()
