# -*- coding: utf-8 -*-
"""E72 (夜航 P6): AERCA 官方实现在 TE 第五覆盖床 (GPU, 官方预算) → 5/5 床.

口径对齐 e69 (SS): 官方配置 (W=1, hidden 1000x4, lr 1e-6, epochs 5000,
causal_quantile 0.70), 训练=loader 正常段, 测试=故障批全程, 窗口级 hit@3
(loader 网格 window=16/stride=60, 根因=ROOT_MAP 工程映射).
输出: _cache/e72_aerca_te.json
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

from data.tep import load_tep_units, BATCH_LEN
from models.aerca import AERCA

CACHE = os.path.join(SRC, '_cache')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(4)
W = 1
WINDOW = 16
STRIDE = 60


def run():
    torch.manual_seed(20260908)
    units = load_tep_units()
    print(f'{len(units)} TE units: {[u["name"] for u in units]}', flush=True)
    out = {}
    for ci, u in enumerate(units):
        t0 = time.time()
        d = u['series_normal'].shape[1]
        mu = u['series_normal'].mean(0)
        sd = u['series_normal'].std(0) + 1e-8
        normal = ((u['series_normal'] - mu) / sd).astype(np.float32)
        full = ((u['series_full'] - mu) / sd).astype(np.float32)
        chunks = [torch.tensor(c, dtype=torch.float32, device=DEVICE)
                  for c in np.array_split(normal, 8)]
        model = AERCA(num_vars=d, hidden_layer_size=1000,
                      num_hidden_layers=4, device=DEVICE, window_size=W,
                      stride=1, lr=1e-6, epochs=5000,
                      causal_quantile=0.70, data_name=f'e72_{u["name"]}')
        model._training(chunks)
        model.eval()
        with torch.no_grad():
            us_n = model._testing_step(
                chunks[0], None, add_u=False)[-1].cpu().numpy()
            mu_u, sd_u = us_n.mean(0), us_n.std(0) + 1e-8
            us = model._testing_step(
                torch.tensor(full, dtype=torch.float32, device=DEVICE),
                None, add_u=False)[-1].cpu().numpy()
        z = -(us - mu_u) / sd_u
        # loader 网格的异常窗右端
        y_onset = WINDOW          # seg 内故障起点近似: 与 loader 同规则
        anom_ends = list(range(max(y_onset + WINDOW, WINDOW),
                               len(z), STRIDE))
        phi = np.stack([z[e - WINDOW:e].mean(0) for e in anom_ends])
        roots = u['R_anom'][0]
        hits = [len(set(np.argsort(-phi[i])[:3]) & roots) / 3
                for i in range(len(phi))]
        hit3 = float(np.mean(hits))
        out[u['name']] = dict(fault=u['fault'], hit3=hit3,
                              n_windows=len(phi),
                              wall_s=round(time.time() - t0, 1))
        json.dump(out, open(os.path.join(CACHE, 'e72_aerca_te.json'), 'w'),
                  default=float, indent=1)
        print(f'[{ci+1}/{len(units)}] {u["name"]}: hit@3={hit3:.3f} '
              f'({out[u["name"]]["wall_s"]}s)', flush=True)
    hits = [v['hit3'] for v in out.values()]
    print(f'\n=== AERCA on TE: mean hit@3 = {np.mean(hits):.3f} '
          f'over {len(hits)} units ===')


if __name__ == '__main__':
    run()
