# -*- coding: utf-8 -*-
"""E44: 外部已发表基线 AERCA (Han et al., ICLR 2025 Oral) 在本文真实测试床上的
窗口级评估.

协议: AERCA 官方实现 (github.com/hanxiao0607/AERCA), MSDS 风格配置
(hidden 1000 x 4, lr 1e-6, 其余默认; epoch 预算 400, 官方早停 patience=20),
正常段前 6000 步切成 8 段训练; 窗口评分 = 编码器潜变量 z 分数
(-(u - mu)/sd, 官方公式) 在窗口内逐变量取最大, 得分排行进本文 hit@K 协议.
测试床: RCAEval 前 3 个故障单元 + SWaT 注入单元.
输出: _cache/e44_aerca.json
"""
import os
import sys
import json
import time
import collections

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, '_external', 'AERCA'))

from data.rcaeval import load_ob_units
from data.swat import build_swat_units
from evaluation import metrics as M
from models.aerca import AERCA

CACHE = os.path.join(SRC, '_cache')
WORKDIR = os.path.join(CACHE, 'aerca_models')
os.makedirs(WORKDIR, exist_ok=True)
os.chdir(WORKDIR)

torch.set_num_threads(8)
DEVICE = torch.device('cpu')
TRAIN_STEPS = 6000
N_CHUNKS = 8
EPOCHS = 400


def train_and_score(name, u):
    d = u['X_anom'].shape[2]
    series = np.asarray(u['series_normal'], dtype=np.float32)[:TRAIN_STEPS]
    mu, sd = series.mean(0), series.std(0) + 1e-8
    series = (series - mu) / sd
    chunks = np.array_split(series, N_CHUNKS)
    xs = [torch.tensor(c, dtype=torch.float32) for c in chunks]

    t0 = time.time()
    model = AERCA(num_vars=d, hidden_layer_size=1000, num_hidden_layers=4,
                  device=DEVICE, window_size=1, stride=1, lr=1e-6,
                  epochs=EPOCHS, causal_quantile=0.70,
                  data_name=f'e44_{name}')
    model._training(xs)
    minutes = (time.time() - t0) / 60

    model.eval()
    us_all = []
    with torch.no_grad():
        for x in xs:
            us_all.append(model._testing_step(x, None, add_u=False)[-1]
                          .cpu().numpy())
    us_all = np.concatenate(us_all, 0)
    mu_u = us_all.mean(0)
    sd_u = us_all.std(0) + 1e-8

    Xa = np.asarray(u['X_anom'], dtype=np.float32)
    Xa = (Xa - mu) / sd
    phi = []
    with torch.no_grad():
        for i in range(len(Xa)):
            x = torch.tensor(Xa[i], dtype=torch.float32)
            us = model._testing_step(x, None, add_u=False)[-1].cpu().numpy()
            z = -(us - mu_u) / sd_u
            phi.append(z.max(0))
    phi = np.stack(phi)
    hit1 = float(M.mean_hit(phi, u['R_anom'], 1))
    hit3 = float(M.mean_hit(phi, u['R_anom'], 3))
    hit5 = float(M.mean_hit(phi, u['R_anom'], 5))
    rec = dict(unit=name, d=int(d), n_windows=int(len(Xa)),
               hit1=hit1, hit3=hit3, hit5=hit5,
               train_minutes=round(minutes, 1),
               epochs_budget=EPOCHS, train_steps=TRAIN_STEPS)
    print(json.dumps(rec), flush=True)
    return rec


def run():
    records = []
    for k, u in enumerate(load_ob_units(reps=(1,))[:3]):
        records.append(train_and_score(f'rcaeval_{k}', u))
    u = build_swat_units()
    records.append(train_and_score('swat_injected', u))
    json.dump(records, open(os.path.join(CACHE, 'e44_aerca.json'), 'w'),
              default=float)
    print('=== AERCA (published baseline) summary ===')
    for r in records:
        print(f"  {r['unit']:16s} d={r['d']:3d} n={r['n_windows']:4d} "
              f"hit@3={r['hit3']:.3f} hit@1={r['hit1']:.3f} "
              f"({r['train_minutes']}min)")


if __name__ == '__main__':
    run()
