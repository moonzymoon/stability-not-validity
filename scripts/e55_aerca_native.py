# -*- coding: utf-8 -*-
"""E55: AERCA 原生 AC@k 协议对照 (SCM 3 seeds, GPU, 官方预算).

目的: 证明窗口级改编既不美化也不贬低 AERCA —— 用官方 episode 级
topk_at_step 指标 (AC@k) 在同一 SCM 数据上评测, 与 e50 窗口级 hit@3 对照.
episode = series[t_start-w : t_start+length], label = (T,d) 0/1,
根因列在异常期内为 1 (与其 SWaT loader 的注入期标签构造一致).
输出: _cache/e55_native.json
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import sys
import json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, '_external', 'AERCA'))

from data.scm import make_sample
from models.aerca import AERCA

CACHE = os.path.join(SRC, '_cache')
WORK = os.path.join(CACHE, 'aerca_models_e55')
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(4)


def topk_at_step(scores, labels, k_range=10):
    k_lst = []
    for i in range(len(labels)):
        if sum(labels[i]) > 0:
            ranking = np.argsort(scores[i])
            label_ind = np.where(labels[i] == 1)[0]
            for k in range(1, k_range + 1):
                count = [1 if j in label_ind else 0 for j in ranking[-k:]]
                k_lst.append(sum(count) / min(k, len(label_ind)))
    return np.array(k_lst).reshape(-1, k_range).mean(axis=0)


def run():
    W = 1  # AERCA window_size=1 (官方 SWaT/MSDS 配置)
    out = {}
    for seed in range(3):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2, seed=seed,
                        noise_std=0.5, magnitude=0.5)
        d = s['n_nodes']
        normal = np.asarray(s['normal'], np.float32)[:6000]
        mu, sd = normal.mean(0), normal.std(0) + 1e-8
        chunks = [torch.tensor((c - mu) / sd, dtype=torch.float32,
                               device=DEVICE)
                  for c in np.array_split((normal - mu) / sd, 8)]
        model = AERCA(num_vars=d, hidden_layer_size=1000, num_hidden_layers=4,
                      device=DEVICE, window_size=W, stride=1, lr=1e-6,
                      epochs=5000, causal_quantile=0.70,
                      data_name=f'e55_scm{seed}')
        model._training(chunks)
        model.eval()
        us_all = []
        with torch.no_grad():
            for x in chunks:
                us_all.append(model._testing_step(x, None, add_u=False)[-1]
                              .cpu().numpy())
        us_all = np.concatenate(us_all, 0)
        mu_u, sd_u = us_all.mean(0), us_all.std(0) + 1e-8

        acs = []
        for a in s['anomalies']:
            t0, L = a['t_start'], a['length']
            seg = s['series'][max(0, t0 - 2 * W): t0 + L]
            lab = np.zeros((len(seg), d))
            roots = list(a['roots'])
            lab[2 * W:, roots] = 1
            x = torch.tensor((seg - mu) / sd, dtype=torch.float32,
                             device=DEVICE)
            with torch.no_grad():
                us = model._testing_step(x, None, add_u=False)[-1] \
                    .cpu().numpy()
            z = -(us - mu_u) / sd_u
            zs = z[W:]
            ls = lab[2 * W:]
            n = min(len(zs), len(ls))
            acs.append(topk_at_step(zs[:n], ls[:n]))
        acs = np.stack(acs).mean(0)
        out[f'scm{seed}'] = dict(ac1=float(acs[0]), ac3=float(acs[2]),
                                 ac5=float(acs[4]),
                                 n_episodes=len(s['anomalies']))
        print(f'scm{seed}: AC@1={acs[0]:.3f} AC@3={acs[2]:.3f} '
              f'AC@5={acs[4]:.3f}', flush=True)
    json.dump(out, open(os.path.join(CACHE, 'e55_native.json'), 'w'))
    print('-> e55_native.json')


if __name__ == '__main__':
    run()
