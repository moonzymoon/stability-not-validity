# -*- coding: utf-8 -*-
"""E66 (修订弹药 B): 第五测试床 RCAEval-SS (Sock Shop) 双指标评测.

协议**完全对齐 e58_rcaeval_acr** (审稿人会与四床结果并排比较):
  单元   全部 (service×fault) × reps 1-3
  图     pcmci_clean (正常段) / pcmci_anom (全程), PCMCI tau_max=1,
         图缓存 _cache/e66_graph_{unit}.npz; GRAA 边置信度沿用与 e58 相同的
         注入 (两图源共用污染图 val, 逐行对齐)
  方法   PropRank / DPTA-G / GRAA(p=0.4) → 图扰动 ACR@3 (del/add/rew 循环,
         q=0.1, M=8) + hit@3; zDev → hit@3 (图无关, 无图 ACR)
  汇总   稳定-错误份额 (cell 级 ACR>0.8 & hit<0.4) 按方法
AERCA: 跳过 (GPU 预算外, 与 e58 口径一致的说明性 omission)。
输出: _cache/e66_rcaeval2.json (每 5 单元增量落盘一次, 防中断丢全部)
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
import sys
import json
import time
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.rcaeval2 import load_units
from graphs.sources import pcmci_graph
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graph_free import GRAPH_FREE
from attribution.graa_v4 import graa_v4_attribute
from scripts.e1_anchor import perturb_graph, stable_seed
from evaluation import metrics as M

CACHE = os.path.join(SRC, '_cache')
Q = 0.1
M_INST = 8
OUT = os.path.join(CACHE, 'e66_rcaeval2.json')


def graph_with_val(series, cache_name):
    path = os.path.join(CACHE, cache_name)
    if os.path.exists(path):
        d = np.load(path)
        return d['A'], d['val']
    r = pcmci_graph(series, tau_max=1)
    val = np.abs(r['val_matrix']).max(2)
    val = val / (val.max() + 1e-9)
    np.savez(path, A=r['adj'], val=val)
    return r['adj'], val


def acr(phi0, phis, K=3):
    t0 = np.argsort(-phi0, axis=1)[:, :K]
    ov = []
    for pm in phis:
        tm = np.argsort(-pm, axis=1)[:, :K]
        ov.append(np.mean([len(set(a) & set(b)) / K
                           for a, b in zip(t0, tm)]))
    return float(np.mean(ov))


def run(mode='ss'):
    """mode: 'ss' = Sock Shop (目标); 'ob_holdout' = OB rep4-5 未见过重复 (降级)."""
    t_all = time.time()
    if mode == 'ss':
        units = load_units('ss', reps=(1, 2, 3))
        out_name = OUT
    else:
        from data.rcaeval import load_ob_units
        units = load_ob_units(reps=(4, 5))
        # 严格对齐 e58 的故障构成 (cpu/mem), 排除组合混杂
        units = [u for u in units if u['fault'] in ('cpu', 'mem')]
        out_name = os.path.join(CACHE, 'e66_ob_holdout.json')
    print(f'mode={mode}: {len(units)} units', flush=True)
    n0 = len(units)
    units = [u for u in units
             if not (np.isnan(u['series_full']).any()
                     or np.isnan(u['series_normal']).any())]
    if len(units) < n0:
        print(f'  dropped {n0 - len(units)} NaN units (tigramite 拒收 NaN;'
              f' 剩余 {len(units)})', flush=True)
    if len(units) < 10:
        print('!! units < 10 — 计划验收边界, 结果须注明样本量限制')
    records = []
    for ui, u in enumerate(units):
        t0 = time.time()
        A_clean, val_clean = graph_with_val(
            u['series_normal'], f"e66_graph_clean_{u['name']}.npz")
        A_anom, val_anom = graph_with_val(
            u['series_full'], f"e66_graph_anom_{u['name']}.npz")
        scorer = make_scorer('iforest').fit(u['X_pool'])
        ctx0 = Context(u['X_pool'], scorer=scorer)
        for gname, A, val in (('pcmci_clean', A_clean, val_anom),
                              ('pcmci_anom', A_anom, val_anom)):
            ctx_g = ctx0.with_graph(A)
            jobs = [('PropRank', lambda: GRAPH_DEPENDENT['PropRank'](
                u['X_anom'], ctx_g)),
                    ('DPTA-G', lambda: GRAPH_DEPENDENT['DPTA-G'](
                        u['X_anom'], ctx_g)),
                    ('GRAA(p=0.4)', lambda: graa_v4_attribute(
                        u['X_anom'], ctx_g, edge_conf=val,
                        prune_frac=0.4)[0])]
            for mname, base_fn in jobs:
                phi0 = base_fn()
                phis = []
                for mi in range(M_INST):
                    rng = np.random.default_rng(stable_seed(
                        'e66', u['name'], gname, mname, mi))
                    A_m = perturb_graph(A, ['del', 'add', 'rew'][mi % 3],
                                        Q, rng)
                    ctx_m = ctx0.with_graph(A_m)
                    if mname.startswith('GRAA'):
                        phis.append(graa_v4_attribute(
                            u['X_anom'], ctx_m, edge_conf=val,
                            prune_frac=0.4)[0])
                    else:
                        phis.append(GRAPH_DEPENDENT[mname](
                            u['X_anom'], ctx_m))
                records.append(dict(
                    unit=u['name'], method=mname, graph_source=gname,
                    hit3=float(M.mean_hit(phi0, u['R_anom'], 3)),
                    acr3=acr(phi0, phis)))
            # zDev: hit@3 only (图无关)
            zd = GRAPH_FREE['zDev'](u['X_anom'], ctx0)
            records.append(dict(
                unit=u['name'], method='zDev', graph_source=gname,
                hit3=float(M.mean_hit(zd, u['R_anom'], 3)), acr3=None))
        records.append(dict(unit=u['name'], method='_meta', wall_s=round(
            time.time() - t0, 1)))
        print(f"[{ui+1}/{len(units)}] {u['name']} "
              + ' '.join(f"{r['method']}={r['acr3']:.2f}"
                         if r['acr3'] is not None
                         else f"{r['method']}:hit={r['hit3']:.2f}"
                         for r in records[-4:] if r['method'] != '_meta'),
              f"({records[-1]['wall_s']}s)", flush=True)
        if (ui + 1) % 5 == 0:
            json.dump(records, open(out_name + '.part', 'w'), default=float)
    json.dump(records, open(out_name, 'w'), default=float)
    if os.path.exists(out_name + '.part'):
        os.remove(out_name + '.part')

    agg = collections.defaultdict(list)
    for r in records:
        if r['method'] == '_meta':
            continue
        agg[(r['method'], r['graph_source'])].append(
            (r['acr3'], r['hit3']))
    print('\n=== E66 SS ACR@3/hit@3 (q=0.1, M=8) ===')
    for k in sorted(agg, key=str):
        v = agg[k]
        acrs = [x[0] for x in v if x[0] is not None]
        hit3 = np.mean([x[1] for x in v])
        astr = f'acr3={np.mean(acrs):.3f} ' if acrs else 'acr3=  --  '
        print(f'  {str(k):30} {astr}hit3={hit3:.3f} (n={len(v)})')
    print('\n=== 稳定-错误份额 (cell 级, ACR>0.8 & hit<0.4) ===')
    for m in ('PropRank', 'DPTA-G', 'GRAA(p=0.4)'):
        cells = [x for k, v in agg.items() if k[0] == m for x in v
                 if x[0] is not None]
        sw = [1.0 if (a > 0.8 and h < 0.4) else 0.0 for a, h in cells]
        print(f'  {m:14} {np.mean(sw)*100:.1f}%  (n={len(sw)})')
    print(f'total {time.time()-t_all:.0f}s -> {out_name}')


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else 'ss')
