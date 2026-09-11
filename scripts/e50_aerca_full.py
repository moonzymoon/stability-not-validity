# -*- coding: utf-8 -*-
"""E50: AERCA 官方满协议 (GPU, epochs=5000 官方早停) 扩展覆盖 + 双指标.

单元: RCAEval 前6 + SWaT + SCM seeds 0-2 (共10).
效度: hit@1/3/5 (窗口级, z-max 排行);
稳定度: 输入加噪扰动 (q=0.05/0.10 通道std) 下 top-3 重叠 ACR@3 —— 外部方法
首次进入双指标协议. 输出: _cache/e50_aerca_full.json
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import sys
import json
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, '_external', 'AERCA'))

from data.rcaeval import load_ob_units
from data.swat import build_swat_units
from data.scm import make_sample, build_windows
from evaluation import metrics as M
from models.aerca import AERCA

CACHE = os.path.join(SRC, '_cache')
WORK = os.path.join(CACHE, 'aerca_models_e50')
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(4)
EPOCHS = 5000
TRAIN_STEPS = 6000
QS = (0.05, 0.10)


def units():
    for k, u in enumerate(load_ob_units(reps=(1,))[:6]):
        yield f'rcaeval_{k}', u
    yield 'swat', build_swat_units()
    for seed in range(3):
        s = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                        seed=seed, noise_std=0.5, magnitude=0.5)
        w = build_windows(s)
        w = dict(w)
        w['series_normal'] = s['normal']
        yield f'scm{seed}', w


def run():
    records = []
    for name, u in units():
        d = u['X_anom'].shape[2]
        series = np.asarray(u['series_normal'], np.float32)[:TRAIN_STEPS]
        mu, sd = series.mean(0), series.std(0) + 1e-8
        series = (series - mu) / sd
        chunks = [torch.tensor(c, dtype=torch.float32, device=DEVICE)
                  for c in np.array_split(series, 8)]
        t0 = time.time()
        model = AERCA(num_vars=d, hidden_layer_size=1000, num_hidden_layers=4,
                      device=DEVICE, window_size=1, stride=1, lr=1e-6,
                      epochs=EPOCHS, causal_quantile=0.70,
                      data_name=f'e50_{name}')
        model._training(chunks)
        minutes = (time.time() - t0) / 60
        model.eval()
        us_all = []
        with torch.no_grad():
            for x in chunks:
                us_all.append(model._testing_step(x, None, add_u=False)[-1]
                              .cpu().numpy())
        us_all = np.concatenate(us_all, 0)
        mu_u, sd_u = us_all.mean(0), us_all.std(0) + 1e-8

        def score(Xw):
            Xn = (Xw - mu) / sd
            phi = []
            with torch.no_grad():
                for i in range(len(Xn)):
                    x = torch.tensor(Xn[i], dtype=torch.float32,
                                     device=DEVICE)
                    us = model._testing_step(x, None, add_u=False)[-1] \
                        .cpu().numpy()
                    phi.append((-(us - mu_u) / sd_u).max(0))
            return np.stack(phi)

        Xa = np.asarray(u['X_anom'], np.float32)
        phi0 = score(Xa)
        rec = dict(unit=name, d=int(d), n_windows=int(len(Xa)),
                   hit1=float(M.mean_hit(phi0, u['R_anom'], 1)),
                   hit3=float(M.mean_hit(phi0, u['R_anom'], 3)),
                   hit5=float(M.mean_hit(phi0, u['R_anom'], 5)),
                   minutes=round(minutes, 1), device=str(DEVICE),
                   epochs=EPOCHS)
        for q in QS:
            rng = np.random.default_rng(50)
            Xp = Xa + rng.normal(0, q * Xa.std(), Xa.shape).astype(np.float32)
            phip = score(Xp)
            rec[f'acr3_q{q}'] = float(
                np.mean([M.acr_at_k(phip, phi0, 3)[i] for i in
                         range(min(len(phi0), len(phip)))]))
        records.append(rec)
        print(json.dumps(rec), flush=True)
        json.dump(records, open(os.path.join(CACHE, 'e50_aerca_full.json'),
                                'w'), default=float)
    print('-> e50_aerca_full.json')


if __name__ == '__main__':
    run()
