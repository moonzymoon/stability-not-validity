# -*- coding: utf-8 -*-
"""E18: GRAA方法评估 — 与所有基线在三个图源+三个测试床上比较."""
import os, sys, json, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.scm import make_sample, build_windows
from graphs.sources import pcmci_graph
from graphs import graph_ops as go
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graph_free import GRAPH_FREE
from attribution.graa import grcm_attribute
from evaluation import metrics as M
from scripts.e8_round6 import gcn_rank_attribute
from scripts.e1_anchor import perturb_graph, stable_seed, M_PERTURB

CACHE = os.path.join(SRC, '_cache')


def run_scm():
    """SCM: GRAA vs 所有基线 × 3图源."""
    records = []
    for seed in range(5):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                        seed=seed, noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        A_true = s['adjacency'].astype(float)
        A_pcmci = pcmci_graph(s['normal'])['adj']
        A_anom = pcmci_graph(s['series'])['adj']
        graphs = {'true': A_true, 'pcmci': A_pcmci, 'pcmci_anom': A_anom}

        # 边置信度: |PCMCI val|
        val_c = np.abs(pcmci_graph(s['normal'])['val_matrix']).max(2)
        # 归一化到[0,1]
        val_c = val_c / (val_c.max() + 1e-9)

        scorer = make_scorer('iforest').fit(w['X_pool'])
        ctx0 = Context(w['X_pool'], scorer=scorer)

        for gname, A in graphs.items():
            ctx_g = ctx0.with_graph(A)

            # GRAA
            phi_graa, info = grcm_attribute(w['X_anom'], ctx_g,
                                            edge_conf=val_c, alpha_base=0.5)
            h1 = M.mean_hit(phi_graa, w['R_anom'], 1)
            h3 = M.mean_hit(phi_graa, w['R_anom'], 3)
            h5 = M.mean_hit(phi_graa, w['R_anom'], 5)

            # ACR
            acrs = []
            for mi in range(M_PERTURB):
                rng = np.random.default_rng(stable_seed(
                    'e18', seed, 'GRAA', gname, mi))
                A_m = perturb_graph(A, ['del','add','rew'][mi%3], 0.2, rng)
                ctx_m = ctx0.with_graph(A_m)
                phi_m, _ = grcm_attribute(w['X_anom'], ctx_m,
                                         edge_conf=val_c, alpha_base=0.5)
                acrs.append(M.acr_at_k(phi_m, phi_graa, 3).mean())

            records.append(dict(
                dataset=f'scm{seed}', method='GRAA', graph_source=gname,
                hit1=h1, hit3=h3, hit5=h5, acr3=float(np.mean(acrs)),
                alpha_mean=info['alpha_mean'], cons_mean=info['consistency_mean']))
            print(f'  scm{seed} GRAA {gname}: hit@3={h3:.3f} ACR={np.mean(acrs):.3f} α={info["alpha_mean"]:.2f}',
                  flush=True)

            # 基线
            for mname, fn in list(GRAPH_DEPENDENT.items()):
                phi = fn(w['X_anom'], ctx_g)
                acrs_b = []
                for mi in range(4):
                    rng = np.random.default_rng(stable_seed(
                        'e18b', seed, mname, gname, mi))
                    A_m = perturb_graph(A, ['del','add','rew'][mi%3], 0.2, rng)
                    phi_m = fn(w['X_anom'], ctx0.with_graph(A_m))
                    acrs_b.append(M.acr_at_k(phi_m, phi, 3).mean())
                records.append(dict(
                    dataset=f'scm{seed}', method=mname, graph_source=gname,
                    hit1=M.mean_hit(phi, w['R_anom'], 1),
                    hit3=M.mean_hit(phi, w['R_anom'], 3),
                    hit5=M.mean_hit(phi, w['R_anom'], 5),
                    acr3=float(np.mean(acrs_b))))

            for mname in ('zDev', 'AERec'):
                fn = GRAPH_FREE[mname]
                phi = fn(w['X_anom'], ctx0)
                records.append(dict(
                    dataset=f'scm{seed}', method=mname, graph_source='none',
                    hit1=M.mean_hit(phi, w['R_anom'], 1),
                    hit3=M.mean_hit(phi, w['R_anom'], 3),
                    hit5=M.mean_hit(phi, w['R_anom'], 5), acr3=None))

    json.dump(records, open(os.path.join(CACHE, 'e18_graa.json'), 'w'), default=float)

    # 汇总
    agg = collections.defaultdict(lambda: dict(h=[], a=[]))
    for r in records:
        agg[(r['method'], r['graph_source'])]['h'].append(r['hit3'])
        if r.get('acr3') is not None:
            agg[(r['method'], r['graph_source'])]['a'].append(r['acr3'])
    print('\n=== GRAA vs baselines (SCM medium, hit@3 / ACR@3) ===')
    for k in sorted(agg, key=str):
        h = np.mean(agg[k]['h'])
        a = np.mean(agg[k]['a']) if agg[k]['a'] else float('nan')
        print(f'  {str(k):38s} hit={h:.3f} acr={a:.3f}')
    return records


if __name__ == '__main__':
    run_scm()
