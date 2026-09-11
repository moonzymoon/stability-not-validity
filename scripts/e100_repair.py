# -*- coding: utf-8 -*-
"""E100: 干预式效度轴 (措施一) --- repair score on a propagating nonlinear SCM.

三轨仿真(同一固定噪声): 注入轨 / 反事实无注入轨 / 修理轨(窗口内逐部把
归因 top-k 变量钳回反事实值再前向传播). repair@k = (s0 - s_repaired)/
(s0 - s_normal). 对照: oracle 钳真根 / random 钳随机变量.
预期: 修理真根 -> 后代恢复 -> 警报消; 修理后代 -> 根仍在注入 -> 警报存;
修理无关变量 -> 无变化. hit@K 说"指对了人", repair 说"修对了能停警".
-> _cache/e100_repair.json
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graph_free import GRAPH_FREE
from evaluation import metrics as M
from scripts.e93_nonlinear import make_dag

CACHE = os.path.join(SRC, '_cache')
D, T, WIN, STRIDE, NOISE = 15, 6000, 16, 20, 0.5
MAGS = (0.6, 0.9, 1.2)
N_EP = 5
N_SEEDS = 5


def sim(W, noise, inj_node=None, inj_t0=0, inj_dur=0,
        clamp_vars=None, clamp_t0=0, clamp_dur=0, cf=None):
    X = np.zeros((T, D), dtype=np.float32)
    for t in range(T - 1):
        X[t + 1] = np.tanh(W @ X[t]) + noise[t]
        # 注入: 仿真内部逐部 (真传播)
        if inj_node is not None and inj_t0 <= t + 1 < inj_t0 + inj_dur:
            X[t + 1, inj_node] += MAG_CUR
        # 修理: 把 clamp_vars 钳到反事实值 (do-干预)
        if clamp_vars is not None and cf is not None \
                and clamp_t0 <= t + 1 < clamp_t0 + clamp_dur:
            X[t + 1, list(clamp_vars)] = cf[t + 1, list(clamp_vars)]
    return X


MAG_CUR = 1.2


def main():
    out = {'per_mag': {}}
    for mag in MAGS:
        global MAG_CUR
        MAG_CUR = mag
        run_one_mag(out, mag)
    json.dump(out, open(os.path.join(CACHE, 'e100_repair.json'), 'w'),
              indent=1)
    print('-> e100_repair.json')


def run_one_mag(out, mag):
    acc = {}
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed + 11)
        A_bin, order = make_dag(seed)
        W = 0.6 * A_bin / np.maximum(A_bin.sum(0, keepdims=True), 1)
        noise = rng.normal(0, NOISE, (T - 1, D)).astype(np.float32)
        cf = sim(W, noise)                      # 反事实无注入轨

        def sim_inj(node, t0):
            return sim(W, noise, inj_node=node, inj_t0=t0, inj_dur=WIN * STRIDE)

        # 池: 反事实轨正常窗
        starts = rng.choice(T - WIN - 1, 300, replace=False)
        X_pool = np.stack([cf[s:s + WIN] for s in starts]).astype(np.float32)
        scorer = make_scorer('iforest').fit(X_pool)
        ctx0 = Context(X_pool, scorer=scorer)

        for e in range(N_EP):
            node = int(order[e])
            t0 = T // 2 + e * (WIN * STRIDE)
            Xinj = sim_inj(node, t0)
            windows = [Xinj[t0 + k * STRIDE: t0 + k * STRIDE + WIN]
                       for k in range(WIN)]
            # 只取前 8 个窗口控制成本
            windows = windows[:8]
            X_anom = np.stack(windows).astype(np.float32)
            roots = [ {node} ] * len(X_anom)
            for mname, fn in (('PropRank', GRAPH_DEPENDENT['PropRank']),
                              ('DPTA-G', GRAPH_DEPENDENT['DPTA-G']),
                              ('GraphGranger',
                               GRAPH_DEPENDENT['GraphGranger']),
                              ('zDev', GRAPH_FREE['zDev']),
                              ('AERec', GRAPH_FREE['AERec']),
                              ('Random', GRAPH_FREE['Random'])):
                if mname in GRAPH_DEPENDENT:
                    phi = fn(X_anom, ctx0.with_graph(A_bin))
                else:
                    phi = fn(X_anom, ctx0)
                h3 = M.hit_at_k(phi, roots, 3)
                s0 = scorer.score(X_anom)
                # 每窗修理 top-3 (或 oracle/random)
                for i in range(len(X_anom)):
                    w0_abs = t0 + i * 0  # 窗口在注入轨中的绝对位置
                    top3 = set(np.argsort(-phi[i])[:3].tolist())
                    oracle = {node}
                    rnd = set(
                        (np.random.default_rng(seed * 100 + e * 10 + i)
                         .choice(D, 3, replace=False)).tolist())
                    for tag, vars_ in (('top3', top3), ('oracle', oracle),
                                       ('random', rnd)):
                        wt0 = t0 + i * STRIDE
                        Xr = sim(W, noise, inj_node=node, inj_t0=t0,
                                 inj_dur=WIN * STRIDE,
                                 clamp_vars=vars_, clamp_t0=wt0,
                                 clamp_dur=WIN, cf=cf)
                        wrep = Xr[wt0:wt0 + WIN][None].astype(np.float32)
                        srep = scorer.score(wrep)[0]
                        snorm = scorer.score(
                            cf[wt0:wt0 + WIN][None].astype(np.float32))[0]
                        s0i = scorer.score(
                            X_anom[i:i + 1].astype(np.float32))[0]
                        rep = (s0i - srep) / (abs(s0i - snorm) + 1e-9)
                        acc.setdefault(mname, {}).setdefault(
                            tag, []).append(float(rep))
                    acc[mname].setdefault('hit', []).append(float(h3[i]))
        print(f'seed {seed} done', flush=True)

    out['per_mag'][str(mag)] = {
        m: {k: float(np.mean(v)) for k, v in d.items()}
        for m, d in acc.items()}
    print(f'=== MAG {mag} ===', flush=True)
    for m, d in out['per_mag'][str(mag)].items():
        print(f'{m:12s}', {k: round(v, 3) for k, v in d.items()}, flush=True)


if __name__ == '__main__':
    main()
