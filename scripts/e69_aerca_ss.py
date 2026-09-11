# -*- coding: utf-8 -*-
"""E69 (夜航 P3): AERCA 官方实现在 SS 第五测试床 (GPU, 窗口级 hit@3).

对齐 e50/e58 的 RCAEval AERCA 口径 (官方实现+官方预算, 窗口级 hit@3),
把 Limitations 的 "AERCA enters on three of the five testbeds" 推进到四个.
案例: SS 每服务取 1 例 (优先 cpu/mem), rep 1, 共 6 例。
协议: W=1, hidden 1000 x4, lr 1e-6, epochs 5000, causal_quantile 0.70
(官方 SWaT/MSDS 配置, 与 e55 完全一致); 训练=注入前正常段; 测试=全程;
窗口聚合: 对 loader 异常窗 (t_end 网格) 取窗内均分 → hit@3 vs 根因列集。
输出: _cache/e69_aerca_ss.json
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

from data.rcaeval2 import load_units
from models.aerca import AERCA

CACHE = os.path.join(SRC, '_cache')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(4)
W = 1


def window_hit(us, u, on):
    """us: (T, D) 逐步偏差分 → 每异常窗取均分 → hit@3."""
    ends = list(range(30, on, 13))  # 占位(未用池), 异常窗如下
    anom_ends = list(range(on + 30, len(us), 30))
    phi = np.stack([us[e - 30:e].mean(0) for e in anom_ends])
    hits = []
    for i, e in enumerate(anom_ends):
        top = set(np.argsort(-phi[i])[:3])
        roots = u['R_anom'][0]
        hits.append(len(top & roots) / 3)
    return float(np.mean(hits)), len(anom_ends), phi


def run():
    torch.manual_seed(20260907)
    units = load_units('ss', reps=(1,))
    # 每服务取 1 例, 优先 cpu/mem
    by_svc = {}
    for u in units:
        svc = u['service']
        prefer = u['fault'] in ('cpu', 'mem')
        if svc not in by_svc or (prefer and by_svc[svc][1] not in
                                 ('cpu', 'mem')):
            by_svc[svc] = (u, u['fault'])
    picked = [by_svc[s][0] for s in sorted(by_svc)][:6]
    print(f'{len(picked)} cases: {[u["name"] for u in picked]}', flush=True)

    out = {}
    for ci, u in enumerate(picked):
        t0 = time.time()
        d = len(u['cols'])
        on = len(u['series_normal'])
        mu = u['series_normal'].mean(0)
        sd = u['series_normal'].std(0) + 1e-8
        normal = (u['series_normal'] - mu) / sd
        full = (u['series_full'] - mu) / sd
        chunks = [torch.tensor(c, dtype=torch.float32, device=DEVICE)
                  for c in np.array_split(normal, 8)]
        model = AERCA(num_vars=d, hidden_layer_size=1000,
                      num_hidden_layers=4, device=DEVICE, window_size=W,
                      stride=1, lr=1e-6, epochs=5000,
                      causal_quantile=0.70, data_name=f'e69_{u["name"]}')
        model._training(chunks)
        model.eval()
        with torch.no_grad():
            us = model._testing_step(
                torch.tensor(full, dtype=torch.float32, device=DEVICE),
                None, add_u=False)[-1].cpu().numpy()
        hit3, n_win, _ = window_hit(us, u, on)
        out[u['name']] = dict(service=u['service'], fault=u['fault'],
                              hit3=hit3, n_windows=n_win,
                              wall_s=round(time.time() - t0, 1))
        json.dump(out, open(os.path.join(CACHE, 'e69_aerca_ss.json'),
                            'w'), default=float, indent=1)
        print(f'[{ci+1}/{len(picked)}] {u["name"]}: hit@3={hit3:.3f} '
              f'({out[u["name"]]["wall_s"]}s)', flush=True)
    hits = [v['hit3'] for v in out.values()]
    print(f'\n=== AERCA on SS: mean hit@3 = {np.mean(hits):.3f} '
          f'over {len(hits)} cases ===')


if __name__ == '__main__':
    run()
