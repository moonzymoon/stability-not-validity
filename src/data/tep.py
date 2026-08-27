# -*- coding: utf-8 -*-
"""TE 过程数据适配器 (hit@K 金标准第二来源)。

来源: 复制自第9篇 D:/0科研/工作1/第9篇SCI/src/datasets/tep.py 的数据缓存
(tep_data.npz, Rieth et al. 扩展 TE 仿真, github.com/mv-per/tennessee-eastman-dataset),
本篇重写窗口化与根因映射。

根因映射 (工程知识, Downs & Vogel 1993 过程拓扑 + Rieth et al. 2017):
  IDV(1)  A/C 进料比值阶跃 (stream 4 组成)   -> 直接受影响测量 XMEAS_4  (idx 3)
  IDV(4)  反应器冷却水入口温度阶跃           -> XMEAS_19 反应器冷却水出口温度 (idx 18)
  IDV(11) 反应器冷却水入口温度随机变化       -> 同 IDV(4) (idx 18)
诚实声明: TE 的变量级根因是"直接受影响测量"的工程映射而非受控注入, 与 SCM 的
构造性真值不同层级 —— 论文中作为真实数据佐证并明示此限制 (第10篇诚实报告范式)。
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(os.path.dirname(HERE), '_cache', 'tep_data.npz')

# 每个故障运行 50h, 前 8h 正常; dt=60s -> 3001 样本/批
FAULT_ONSET = 8 * 60          # 采样步
BATCH_LEN = 3001
UNITS = [('idv1', 0), ('idv1', 1), ('idv4', 0), ('idv4', 1),
         ('idv11', 0), ('idv11', 1)]
ROOT_MAP = {'idv1': {3}, 'idv4': {18}, 'idv11': {18}}   # XMEAS 0-indexed


def load_tep_units(window: int = 16, stride: int = 60, pool_stride: int = 13):
    """返回 list[unit dict]: 每个 TE 故障批一个评估单元。

    X_anom: 故障段窗口 (t >= onset); X_pool: 全局正常段 (normal_500 前 35%)
    R_anom: 每窗口根因集 (ROOT_MAP); series_normal: 学图用正常段。
    """
    d = np.load(NPZ)
    X, Y = d['X'], d['Y']
    n_normal = 30001                      # normal_500 段长度
    Xn = X[:n_normal]
    train_end = int(n_normal * 0.35)
    Xtrain = Xn[:train_end]

    def win(series, t_end):
        return series[t_end - window:t_end]

    # 正常池 (全局)
    ok = range(window, train_end, pool_stride)
    X_pool = np.stack([win(Xn, t) for t in ok]).astype(np.float32)

    units = []
    for k, (fault, batch) in enumerate(UNITS):
        seg = X[n_normal + k * BATCH_LEN: n_normal + (k + 1) * BATCH_LEN]
        yseg = Y[n_normal + k * BATCH_LEN: n_normal + (k + 1) * BATCH_LEN]
        on = np.where(yseg == 1)[0]
        if len(on) == 0:
            continue
        start = on[0]
        ends = range(max(start + window, window), len(seg), stride)
        Xa = np.stack([win(seg, int(t)) for t in ends]).astype(np.float32)
        if len(Xa) == 0:
            continue
        units.append(dict(
            name=f'te_{fault}_b{batch}', fault=fault,
            X_anom=Xa, X_pool=X_pool,
            R_anom=[set(ROOT_MAP[fault])] * len(Xa),
            series_normal=Xtrain,           # PCMCI 学图输入
            series_full=seg,
        ))
    return units


if __name__ == '__main__':
    us = load_tep_units()
    for u in us:
        print(u['name'], 'anom windows:', len(u['X_anom']),
              'roots:', u['R_anom'][0], 'pool:', len(u['X_pool']))
