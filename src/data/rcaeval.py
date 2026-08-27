# -*- coding: utf-8 -*-
"""RCAEval RE1-OB (Online Boutique) 适配器 — 第四测试床: 真实分布式系统 +
注入式构造真值 (根因 = 被注入服务的指标集).

数据: Zenodo 14590730 RE1-OB.zip (CC BY 4.0, Pham et al. WWW'25 Companion).
结构: RE1-OB/{service}_{fault}/{rep}/data.csv (4201x51, 1s 采样) +
inject_time.txt (Unix 注入时刻, 恰在运行中点).
约定 (诚实声明, 与 SWaT 注入式真值同层级):
  - R_anom = 注入服务的全部指标列 (服务级真值投影到变量级;
    cpu 故障下该服务 cpu 为主指标, mem/latency 等同级受扰).
  - 剔除正常段近常数列 (std<1e-9); z-score 用注入前段统计 (部署场景).
  - 无真值图: PCMCI 学图 (正常段=干净源 / 全程=污染源), tau_max=1.
"""
import os
import re

import numpy as np
import pandas as pd

ROOT = r'D:\0科研\工作1\第11篇SCI\data\RCAEval\RE1-OB'
WINDOW = 30          # 30 s
STRIDE = 30
POOL_STRIDE = 13


def load_ob_units(reps=(1,), services=None):
    """返回 list[unit dict]: 每 (service, fault, rep) 一个评估单元."""
    cases = sorted(d for d in os.listdir(ROOT)
                   if os.path.isdir(os.path.join(ROOT, d)))
    units = []
    for case in cases:
        m = re.match(r'(\w+?)_(cpu|mem|disk|delay|loss)$', case)
        if not m:
            continue
        svc, fault = m.group(1), m.group(2)
        if services and svc not in services:
            continue
        for rep in reps:
            d = os.path.join(ROOT, case, str(rep))
            csv = os.path.join(d, 'data.csv')
            if not os.path.exists(csv):
                continue
            df = pd.read_csv(csv)
            df = df.loc[:, ~df.columns.duplicated()]
            cols = [c for c in df.columns
                    if c != 'time' and not c.startswith('time.')]
            t0 = float(open(os.path.join(d, 'inject_time.txt')).read().strip())
            t = df['time'].values.astype(float)
            on = int(np.searchsorted(t, t0))
            Xn = df[cols].values.astype(np.float64)
            # 剔除正常段近常数列
            sd = Xn[:on].std(0)
            keep = sd > 1e-9
            cols = [c for c, k in zip(cols, keep) if k]
            X = Xn[:, keep]
            # z-score (注入前段统计)
            mu = X[:on].mean(0, keepdims=True)
            st = X[:on].std(0, keepdims=True) + 1e-8
            X = (X - mu) / st
            root = {i for i, c in enumerate(cols)
                    if c.rsplit('_', 1)[0] == svc}
            if not root:
                continue

            def win(seg, ends):
                return np.stack([seg[e - WINDOW:e] for e in ends])

            pool_ends = range(WINDOW, on, POOL_STRIDE)
            anom_ends = range(on + WINDOW, len(X), STRIDE)
            X_pool = win(X, pool_ends).astype(np.float32)
            X_anom = win(X, anom_ends).astype(np.float32)
            if len(X_anom) == 0 or len(X_pool) < 50:
                continue
            units.append(dict(
                name=f'ob_{svc}_{fault}_r{rep}', service=svc, fault=fault,
                X_anom=X_anom, X_pool=X_pool,
                R_anom=[root] * len(X_anom), cols=cols,
                series_normal=X[:on].astype(np.float32),
                series_full=X.astype(np.float32)))
    return units


if __name__ == '__main__':
    us = load_ob_units(reps=(1,))
    print(f'{len(us)} units')
    for u in us[:6]:
        print(u['name'], 'anom:', len(u['X_anom']), 'pool:', len(u['X_pool']),
              'roots:', sorted(u['R_anom'][0]))
