# -*- coding: utf-8 -*-
"""E54: 补两个不同机理的失败案例 (SWaT).

case2 伪稳定性: DPTA-G 在 silver 图上 ACR@3=1 且 hit@3=0 的窗口
              (回归路径几乎不消费图 -> 删边不改序).
case3 源偏差:   PropRank 同一窗口 silver 图 hit=1 而 corr/pcmci 图 hit=0
              且错图内部扰动 ACR=1 (稳定地错, 对错敏感于源).
输出: _cache/e54_cases.json
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.swat import build_swat_units
from graphs.sources import pcmci_graph
from scorers import make_scorer
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from evaluation import metrics as M
from scripts.e8_round6 import corr_graph
from scripts.e1_anchor import perturb_graph, stable_seed

CACHE = os.path.join(SRC, '_cache')
OPS = ('del', 'add', 'rew')


def run():
    u = build_swat_units()
    A_silver = u['silver_adj'].astype(float)
    A_pcmci = pcmci_graph(u['series_normal'])['adj'].astype(float)
    A_corr = corr_graph(u['series_normal'][:50000], thresh=0.3)
    scorer = make_scorer('iforest').fit(u['X_pool'])
    ctx0 = Context(u['X_pool'], scorer=scorer)

    out = {}

    def phi_under(fn, A):
        return fn(u['X_anom'], ctx0.with_graph(A))

    def acr_mean(fn, A, tag, i_want=None):
        phi0 = phi_under(fn, A)
        vals = []
        for k in range(8):
            rng = np.random.default_rng(stable_seed('e54', tag, k))
            A_m = perturb_graph(A, OPS[k % 3], 0.2, rng)
            vals.append(M.acr_at_k(phi_under(fn, A_m), phi0, 3))
        return np.stack(vals).mean(0), phi0

    # case2: DPTA-G 伪稳定性 (silver 图)
    acr_d, phi_d = acr_mean(GRAPH_DEPENDENT['DPTA-G'], A_silver, 'dptag')
    hit_d = M.hit_at_k(phi_d, u['R_anom'], 3)
    i2 = next(i for i in range(len(phi_d)) if acr_d[i] == 1.0 and hit_d[i] == 0)
    order = np.argsort(-phi_d[i2])
    out['case2_pseudo_stability'] = dict(
        method='DPTA-G', graph='silver', window=int(i2),
        true_root=[int(x) for x in u['R_anom'][i2]], acr3=1.0, hit3=0,
        top5=[[int(v), round(float(phi_d[i2, v]), 4)] for v in order[:5]])

    # case3: PropRank 源偏差 (silver 对 / corr 错 且错图内稳定)
    acr_c, phi_c = acr_mean(GRAPH_DEPENDENT['PropRank'], A_corr, 'propcorr')
    phi_s = phi_under(GRAPH_DEPENDENT['PropRank'], A_silver)
    hit_c = M.hit_at_k(phi_c, u['R_anom'], 3)
    hit_s = M.hit_at_k(phi_s, u['R_anom'], 3)
    i3 = next(i for i in range(len(phi_c))
              if hit_s[i] == 1 and hit_c[i] == 0 and acr_c[i] >= 0.75)
    order_c = np.argsort(-phi_c[i3])
    order_s = np.argsort(-phi_s[i3])
    out['case3_source_bias'] = dict(
        method='PropRank', window=int(i3),
        true_root=[int(x) for x in u['R_anom'][i3]],
        silver_top3=[int(v) for v in order_s[:3]],
        corr_top3=[int(v) for v in order_c[:3]],
        corr_top5_scores=[[int(v), round(float(phi_c[i3, v]), 4)]
                          for v in order_c[:5]],
        acr3_within_corr=float(acr_c[i3]), hit_silver=1, hit_corr=0)

    json.dump(out, open(os.path.join(CACHE, 'e54_cases.json'), 'w'),
              default=float)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    run()
