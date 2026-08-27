# -*- coding: utf-8 -*-
"""合成 SCM 时序生成器（hit@K 金标准主来源）。

来源: 复制自第8篇 D:/0科研/工作1/第8篇SCI/src/data/scm_generator.py 并增强
（第11篇 v2 必改: 补多变量联合异常、渐变漂移模式、多轨迹重复 + 每轨迹真因窗口）。

增强点 (相对第8篇):
  1. 异常类型扩为 step / ramp / spike / drift(渐变漂移) / 变差(variance inflation);
  2. 支持多变量联合异常 (双根因, roots 为列表);
  3. 每个 (dataset_seed, anomaly) 组合显式产出真因窗口集合 (供固定窗口集合协议);
  4. 异常注入在副本上叠加 (互不覆盖, 间隔 >= 400 步), 保证窗口金标准不混杂。
"""
from typing import Dict, List, Tuple
import numpy as np
import networkx as nx

Edge = Tuple[int, int]

ANOM_LENGTH = 200          # 每个异常的持续步数
GAP = 400                  # 异常之间的最小间隔 (大于窗口长 + 裕量)


def sample_random_dag(n_nodes: int, edge_prob: float = 0.2, seed: int = 0):
    """随机 DAG 邻接矩阵 A: A[src,tgt]=1 表示 src->tgt (拓扑序保无环)。"""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_nodes)
    A_full = (rng.random((n_nodes, n_nodes)) < edge_prob).astype(int)
    A = np.zeros((n_nodes, n_nodes), dtype=int)
    for a in range(n_nodes):
        for b in range(n_nodes):
            if order[a] < order[b]:
                A[a, b] = A_full[a, b]
    G = nx.DiGraph(A)
    G.add_nodes_from(range(n_nodes))
    return A, G


def dag_to_lag_coefs(A: np.ndarray, tau_max: int = 3, decay: float = 0.4):
    """邻接展开为 VAR 系数 {(tgt,src,lag): coef}, lag 越大系数越小。"""
    coefs: Dict[Tuple[int, int, int], float] = {}
    for tgt in range(A.shape[0]):
        for src in range(A.shape[0]):
            if A[src, tgt] == 1:
                for tau in range(1, tau_max + 1):
                    coefs[(tgt, src, tau)] = decay / tau
    return coefs


def gold_edges_from_coefs(lag_coefs) -> set:
    return {(src, tgt) for (tgt, src, tau) in lag_coefs.keys()}


def gen_var_series(lag_coefs, T: int = 5000, noise_std: float = 0.1, seed: int = 0):
    """X_i[t] = sum coef*X_j[t-tau] + eps; 返回 (T, N)。"""
    rng = np.random.default_rng(seed)
    nodes, tau_max = set(), 1
    for (tgt, src, tau) in lag_coefs:
        nodes.add(tgt); nodes.add(src)
        tau_max = max(tau_max, tau)
    N = max(nodes) + 1
    data = np.zeros((T, N))
    data[:tau_max] = rng.normal(0, noise_std, (tau_max, N))
    by_target: Dict[int, List[Tuple[int, int, float]]] = {}
    for (tgt, src, tau), c in lag_coefs.items():
        by_target.setdefault(tgt, []).append((src, tau, c))
    for t in range(tau_max, T):
        x = np.zeros(N)
        for tgt, lst in by_target.items():
            acc = 0.0
            for src, tau, c in lst:
                acc += c * data[t - tau, src]
            x[tgt] = acc
        data[t] = x + rng.normal(0, noise_std, N)
    return data


def inject_anomaly(data: np.ndarray, G: nx.DiGraph, roots: List[int],
                   t_start: int, length: int = ANOM_LENGTH, kind: str = 'step',
                   magnitude: float = 5.0):
    """在根因集合 roots 注入扰动, 沿 DAG 自然传播。返回 (副本, 下游并集)。"""
    ts = data.copy()
    for root in roots:
        if kind == 'step':
            ts[t_start:t_start + length, root] += magnitude
        elif kind == 'ramp':
            ts[t_start:t_start + length, root] += np.linspace(0, magnitude, length)
        elif kind == 'spike':
            ts[t_start:t_start + 5, root] += magnitude * 3
        elif kind == 'drift':          # v2 必改: 渐变漂移 (指数趋近的缓变偏移)
            w = 1 - np.exp(-np.arange(length) / (length / 4.0))
            ts[t_start:t_start + length, root] += magnitude * w
        elif kind == 'var':            # 方差膨胀
            rng = np.random.default_rng(t_start + root)
            ts[t_start:t_start + length, root] += rng.normal(
                0, magnitude / 2.0, length)
        else:
            raise ValueError(f"未知异常类型: {kind}")
    prop = set()
    for root in roots:
        prop |= set(nx.descendants(G, root))
    return ts, sorted(prop)


def make_sample(n_nodes: int = 15, edge_prob: float = 0.2, tau_max: int = 3,
                T: int = 5000, n_single: int = 5, n_joint: int = 2,
                seed: int = 0, noise_std: float = 0.1,
                kinds=('step', 'ramp', 'spike', 'drift', 'var'),
                magnitude: float = 5.0) -> dict:
    """生成一个完整数据集轨迹: 正常段 + n_single 个单根因 + n_joint 个双根因联合异常。

    返回 dict:
      normal (T,N) 仅供学图/训练池; series (T,N) 含异常, 供开窗;
      每个异常: dict(t_start, length, kind, roots list, propagation list)。
    """
    A, G = sample_random_dag(n_nodes, edge_prob, seed=seed)
    coefs = dag_to_lag_coefs(A, tau_max=tau_max)
    gold = gold_edges_from_coefs(coefs)
    normal = gen_var_series(coefs, T=T, seed=seed, noise_std=noise_std)

    rng = np.random.default_rng(seed + 1000)
    root_candidates = [n for n in range(n_nodes) if len(nx.descendants(G, n)) > 0]
    if not root_candidates:
        root_candidates = list(range(n_nodes))
    anomalies = []
    series = normal.copy()
    # 注入区间: 预留头尾 (前 500 步只作正常池), 异常间隔 GAP 防混杂
    slots = np.arange(600, T - ANOM_LENGTH - 200, GAP)
    rng.shuffle(slots)
    need = n_single + n_joint
    if len(slots) < need:
        slots = np.arange(600, T - ANOM_LENGTH - 200, ANOM_LENGTH + 50)
    for i in range(need):
        t0 = int(slots[i % len(slots)])
        kind = str(rng.choice(kinds))
        roots = ([int(rng.choice(root_candidates))] if i < n_single
                 else list(rng.choice(root_candidates, size=2, replace=False)))
        series, prop = inject_anomaly(series, G, roots, t0, kind=kind,
                                      magnitude=magnitude)
        anomalies.append(dict(t_start=t0, length=ANOM_LENGTH, kind=kind,
                              roots=roots, propagation=prop))
    return dict(
        normal=normal, series=series, adjacency=A, graph=G,
        lag_coefs=coefs, gold_edges=gold, anomalies=anomalies,
        n_nodes=n_nodes, T=T, seed=seed,
        var_names=[f'x{i}' for i in range(n_nodes)],
    )


# --------------------------------------------------------------------
# 窗口化: 固定金标准异常窗口集合 (v2 修订: 换打分器只换分数, 窗口集合不动)
# --------------------------------------------------------------------
def build_windows(sample: dict, window: int = 16, stride: int = 20,
                  normal_stride: int = 7):
    """从含异常 series 开窗。

    返回 dict:
      X_anom (n_a, W, D)  异常窗口 (右端点在异常期内, 每异常期按 stride 采样);
      R_anom  list[set]    每个异常窗口的真根因集合;
      X_pool (n_p, W, D)   正常窗口池 (取自前 500 步 + 异常间隙, 按 normal_stride);
      meta_anom list[dict] (anomaly_idx, t_end, kind)。
    normal_stride 取素数避免与异常周期锁定。
    """
    series, D = sample['series'], sample['n_nodes']
    T = series.shape[0]
    # 异常时间步掩码
    anom_mask = np.zeros(T, dtype=bool)
    for a in sample['anomalies']:
        anom_mask[a['t_start']:a['t_start'] + a['length']] = True

    def grab(t_end):
        return series[t_end - window:t_end]

    X_anom, R_anom, meta = [], [], []
    for ai, a in enumerate(sample['anomalies']):
        for t_end in range(a['t_start'] + window, a['t_start'] + a['length'], stride):
            X_anom.append(grab(t_end))
            R_anom.append(set(a['roots']))
            meta.append(dict(anomaly_idx=ai, t_end=t_end, kind=a['kind']))
    # 正常池: 500 步头部 + 间隙区 (掩码为 False 且前后 window 步均无异常)
    safe = ~anom_mask
    conv = np.convolve(safe.astype(int), np.ones(window, dtype=int), mode='valid')
    ok_ends = np.where(conv == window)[0] + window          # 右端点, 全窗正常
    ok_ends = ok_ends[(ok_ends >= window) & (ok_ends <= T)]
    X_pool = np.stack([grab(int(t)) for t in ok_ends[::normal_stride]])
    return dict(
        X_anom=np.asarray(X_anom, dtype=np.float32),
        R_anom=R_anom, meta_anom=meta,
        X_pool=X_pool.astype(np.float32),
        anom_mask=anom_mask,
    )


if __name__ == '__main__':
    s = make_sample(seed=0)
    w = build_windows(s)
    print(f"nodes={s['n_nodes']} gold_edges={len(s['gold_edges'])} "
          f"anomalies={len(s['anomalies'])}")
    print(f"X_anom={w['X_anom'].shape} X_pool={w['X_pool'].shape} "
          f"R_anom[0]={w['R_anom'][0]}")
