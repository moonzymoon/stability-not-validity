# -*- coding: utf-8 -*-
"""RCAEval RE1 泛化适配器 (修订弹药 B, 供 e66) — 第五测试床: Sock Shop.

对 data/rcaeval.py (RE1-OB 专用) 的泛化, **清洗约定逐条保持一致**:
  - 剔除正常段近常数列 (std<1e-9)
  - z-score 用注入前段统计 (+1e-8)
  - R_anom = 注入服务的全部指标列 (服务级真值投影到变量级, 诚实声明同 SWaT/OB 层级)
  - 无真值图: PCMCI 学图 (正常段=干净源 / 全程=污染源), tau_max=1
  - 窗口: WINDOW=30, STRIDE=30, POOL_STRIDE=13; 池取注入前, 异常窗取注入后
  - 守卫: 无异常窗或池<50 跳过
差异: ROOT 按系统参数化 (ss = Sock Shop, Zenodo 14590730 RE1-SS.zip)。
不改 rcaeval.py (旧脚本依赖)。
"""
import os
import re

import numpy as np
import pandas as pd

WINDOW = 30
STRIDE = 30
POOL_STRIDE = 13

ROOTS = {
    'ob': r'D:\0科研\工作1\第11篇SCI\data\RCAEval\RE1-OB',
    'ss': r'D:\0科研\工作1\第11篇SCI\data\RCAEval\RE1-SS',
}

# SS 专属参数 (OB 不受影响):
#   POOL_STRIDE 5: SS 每单元仅 721 行且注入在正中 (on≈360), 步长 13 只切出
#   ~26 池窗触发 <50 守卫; 5 → ~72 窗。
#   指标筛选: SS 暴露全量 cAdvisor (23 服务×48 族≈437 列), PCMCI 不可行;
#   取六指标标准遥测集 (cpu/memory/net-rx/net-tx/istio-request/istio-error,
#   AIOps 常规特征口径), 并剔除 IP 节点导出器 (非微服务)。诚实声明层级
#   与 OB 相同, 修订信中说明。
SS_POOL_STRIDE = 5
SS_METRIC_SUFFIXES = (
    'container-cpu-usage-seconds-total',
    'container-memory-usage-bytes',
    'container-network-receive-bytes-total',
    'container-network-transmit-bytes-total',
    'istio-request-total',
    'istio-error-total',
)


def _ss_keep(col, svc):
    suf = col.rsplit('_', 1)[1]
    return suf in SS_METRIC_SUFFIXES


def load_units(system: str = 'ss', reps=(1, 2, 3), services=None):
    """返回 list[unit dict], 字段与 rcaeval.load_ob_units 相同 (name 前缀随系统)."""
    root = ROOTS[system]
    if not os.path.isdir(root):
        raise FileNotFoundError(f'{system} 数据目录不存在: {root}')
    tag = {'ob': 'ob', 'ss': 'ss'}[system]
    pool_stride = SS_POOL_STRIDE if system == 'ss' else POOL_STRIDE
    cases = sorted(d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d)))
    units = []
    for case in cases:
        m = re.match(r'(\w+?)_(cpu|mem|disk|delay|loss)$', case)
        if not m:
            continue
        svc, fault = m.group(1), m.group(2)
        if services and svc not in services:
            continue
        for rep in reps:
            d = os.path.join(root, case, str(rep))
            csv = os.path.join(d, 'data.csv')
            if not os.path.exists(csv):
                continue
            df = pd.read_csv(csv)
            df = df.loc[:, ~df.columns.duplicated()]
            cols = [c for c in df.columns
                    if c != 'time' and not c.startswith('time.')]
            if system == 'ss':
                cols = [c for c in cols
                        if not re.match(r'^\d+(\.\d+)*-\d+$', c.rsplit('_', 1)[0])
                        and _ss_keep(c, svc)]
            t0 = float(open(os.path.join(d, 'inject_time.txt')).read().strip())
            t = df['time'].values.astype(float)
            on = int(np.searchsorted(t, t0))
            Xn = df[cols].values.astype(np.float64)
            sd = Xn[:on].std(0)
            keep = sd > 1e-9
            cols = [c for c, k in zip(cols, keep) if k]
            X = Xn[:, keep]
            mu = X[:on].mean(0, keepdims=True)
            st = X[:on].std(0, keepdims=True) + 1e-8
            X = (X - mu) / st
            root_set = {i for i, c in enumerate(cols)
                        if c.rsplit('_', 1)[0] == svc}
            if not root_set:
                continue

            def win(seg, ends):
                return np.stack([seg[e - WINDOW:e] for e in ends])

            pool_ends = range(WINDOW, on, pool_stride)
            anom_ends = range(on + WINDOW, len(X), STRIDE)
            X_pool = win(X, pool_ends).astype(np.float32)
            X_anom = win(X, anom_ends).astype(np.float32)
            if len(X_anom) == 0 or len(X_pool) < 50:
                continue
            units.append(dict(
                name=f'{tag}_{svc}_{fault}_r{rep}', service=svc, fault=fault,
                X_anom=X_anom, X_pool=X_pool,
                R_anom=[root_set] * len(X_anom), cols=cols,
                series_normal=X[:on].astype(np.float32),
                series_full=X.astype(np.float32)))
    return units


if __name__ == '__main__':
    import sys
    sys_name = sys.argv[1] if len(sys.argv) > 1 else 'ss'
    us = load_units(sys_name, reps=(1,))
    print(f'[{sys_name}] {len(us)} units (rep1 only)')
    for u in us[:8]:
        print(u['name'], 'D:', len(u['cols']), 'anom:', len(u['X_anom']),
              'pool:', len(u['X_pool']), 'roots:', sorted(u['R_anom'][0]))
