# -*- coding: utf-8 -*-
"""方案5: SWaT 第三测试床 — 真实数据 + 注入式构造真值 (根因=注入通道).

数据: Anomaly-Transformer 版 SWaT train.npy (1.387M×51, 全正常), 复用第10篇
协议思想 (注入式真值); 银标准 26 边图来自第8篇 run_e10_silver_gold.py
(变量名→索引映射按 SWaT 列序)。
"""
import numpy as np

NPY = r'D:\0科研\工作1\Anomaly-Transformer\dataset\SWaT\SWaT_train.npy'

# SWaT 51 列顺序 (来源: 第8篇 real_loader.load_swat 的 CSV 列名, 已与 AT npy
# 常数列指纹核对一致)
COLS = ['FIT101', 'LIT101', 'MV101', 'P101', 'P102', 'AIT201', 'AIT202',
        'AIT203', 'FIT201', 'MV201', 'P201', 'P202', 'P203', 'P204',
        'P205', 'P206', 'DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302',
        'MV303', 'MV304', 'P301', 'P302', 'AIT401', 'AIT402', 'FIT401',
        'LIT401', 'P401', 'P402', 'P403', 'P404', 'UV401', 'AIT501',
        'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502', 'FIT503',
        'FIT504', 'P501', 'P502', 'PIT501', 'PIT502', 'PIT503', 'FIT601',
        'P601', 'P602', 'P603']
COLIDX = {c: i for i, c in enumerate(COLS)}

SILVER = [
    ("MV101", "FIT101"), ("P101", "FIT101"), ("FIT101", "LIT101"),
    ("FIT101", "FIT201"), ("FIT201", "FIT301"), ("FIT301", "FIT401"),
    ("FIT401", "FIT501"), ("FIT501", "FIT601"),
    ("P203", "AIT202"), ("P205", "AIT202"), ("MV201", "FIT201"),
    ("MV302", "FIT301"), ("MV304", "FIT301"), ("P302", "FIT301"),
    ("FIT301", "LIT301"), ("FIT401", "LIT401"),
    ("FIT501", "PIT501"), ("FIT501", "PIT502"), ("FIT501", "PIT503"),
]


def silver_adj():
    d = 51
    A = np.zeros((d, d))
    for a, b in SILVER:
        if a in COLIDX and b in COLIDX:
            A[COLIDX[a], COLIDX[b]] = 1.0
    return A


def load_swat_pool(T_use=120000, seed=0):
    """取 train 前 T_use 步 (正常), z-score 后返回 (T, 51)."""
    d = np.load(NPY, mmap_mode='r')
    X = np.asarray(d[:T_use], dtype=np.float64)
    mu, sd = X[:T_use // 2].mean(0), X[:T_use // 2].std(0) + 1e-8
    return ((X - mu) / sd).astype(np.float32)


def inject_swat(X, chans, t0, length=200, kind='drift', scale=6.0, seed=0):
    """在通道集 chans 注入; 根因=注入通道 (构造性真值). scale 为该通道池std的倍数."""
    rng = np.random.default_rng(seed)
    Y = X.copy()
    for c in chans:
        if kind == 'drift':
            w = 1 - np.exp(-np.arange(length) / (length / 4.0))
            Y[t0:t0 + length, c] += scale * w
        elif kind == 'stuck':
            Y[t0:t0 + length, c] = X[t0, c]
        elif kind == 'var':
            Y[t0:t0 + length, c] += rng.normal(0, scale / 2, length)
        elif kind == 'step':
            Y[t0:t0 + length, c] += scale
    return Y


def build_swat_units(window=16, stride=20, pool_stride=7, n_anom=24, seed=0):
    """构造 SWaT 注入单元: 正常池 + 多个注入事件 (单/双通道).

    返回 dict(X_anom, X_pool, R_anom(根因集), series_normal, silver_adj)
    """
    X = load_swat_pool(seed=seed)
    T, D = X.shape
    rng = np.random.default_rng(seed + 500)
    # 注入通道: 偏向银标准图中出现的通道 (物理上与其他变量有关联, 传播真实)
    silver_chans = sorted({COLIDX[a] for a, b in SILVER} |
                          {COLIDX[b] for a, b in SILVER})
    silver_chans = [c for c in silver_chans if X[:, c].std() > 1e-6]
    events = []
    Y = X.copy()
    occupied = []
    t_cursor = int(T * 0.55)                    # 注入区: 后 45%
    for k in range(n_anom):
        t0 = t_cursor + k * 1200
        if t0 + 400 > T:
            break
        kind = ['drift', 'stuck', 'var', 'step'][k % 4]
        n_ch = 1 if k % 3 else 2
        chans = list(rng.choice(silver_chans, size=n_ch, replace=False))
        Y = inject_swat(Y, chans, t0, kind=kind, seed=seed * 100 + k)
        events.append(dict(t0=t0, length=200, chans=chans))
        occupied.append((t0, t0 + 200))

    def win(series, t_end):
        return series[t_end - window:t_end]

    X_anom, R_anom = [], []
    for ev in events:
        for t_end in range(ev['t0'] + window, ev['t0'] + ev['length'], stride):
            X_anom.append(win(Y, t_end))
            R_anom.append(set(ev['chans']))
    # 正常池: 前 50% 内安全窗
    safe_end = int(T * 0.5)
    X_pool = np.stack([win(X, int(t)) for t in
                       range(window, safe_end, pool_stride)])
    return dict(X_anom=np.asarray(X_anom, dtype=np.float32),
                X_pool=X_pool.astype(np.float32),
                R_anom=R_anom, series_normal=X[:int(T * 0.5)],
                silver_adj=silver_adj(), n_nodes=D)


if __name__ == '__main__':
    u = build_swat_units()
    print('X_anom', u['X_anom'].shape, 'pool', u['X_pool'].shape,
          'silver edges', int(u['silver_adj'].sum()))
    print('root example', u['R_anom'][0])
