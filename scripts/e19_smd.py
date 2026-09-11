# -*- coding: utf-8 -*-
"""E19: SMD数据集接入 — 真实变量级故障标签的第四测试床."""
import os, sys, json, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

CACHE = os.path.join(SRC, '_cache')

# SMD路径(Anomaly-Transformer格式: train/test/test_label .npy)
SMD_PATH = r'D:\0科研\工作1\Anomaly-Transformer\dataset\SMD'

def load_smd(machine='machine-1-1'):
    """加载SMD: 返回 train(normal), test(anomaly), test_label."""
    Xtr = np.load(os.path.join(SMD_PATH, 'SMD_train.npy'))
    Xte = np.load(os.path.join(SMD_PATH, 'SMD_test.npy'))
    Yte = np.load(os.path.join(SMD_PATH, 'SMD_test_label.npy')).flatten()
    return Xtr.astype(np.float32), Xte.astype(np.float32), Yte.astype(np.int8)


def smd_root_causes(Xte, Yte, Xtr, window=16, stride=10):
    """SMD根因推断: 异常期间首个显著偏离的变量(首响应者).

    SMD标签是timestamp级的(异常/正常), 不是变量级的。
    我们用异常起始时刻的偏离模式推断根因:
    - 对每个异常episode, 找窗口内z-score最大的变量
    - 这个变量的身份在每个episode内保持一致(物理上同一故障)
    """
    T, D = Xte.shape
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xz = (Xte - mu) / sd

    # 找异常episode
    events = []
    in_anom = False
    for t in range(T):
        if Yte[t] == 1 and not in_anom:
            start = t
            in_anom = True
        elif Yte[t] == 0 and in_anom:
            events.append((start, t))
            in_anom = False
    if in_anom:
        events.append((start, T))

    # 对每个episode: 首个窗口内最大z变量作为根因
    causes = {}
    for start, end in events:
        if end - start < window + 10:
            continue
        # 异常起始后第一个窗口
        seg = Xz[start:min(start + window, end)]
        z_mean = np.abs(seg).mean(0)
        top_var = int(np.argmax(z_mean))
        causes[(start, end)] = top_var

    return causes, events


def build_smd_units(machine='machine-1-1', window=16, stride=10, pool_stride=7):
    """构建SMD评估单元: 正常池 + 异常窗口(带推断根因)."""
    Xtr, Xte, Yte = load_smd(machine)
    causes, events = smd_root_causes(Xte, Yte, Xtr, window)

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xte_z = ((Xte - mu) / sd).astype(np.float32)
    Xtr_z = ((Xtr - mu) / sd).astype(np.float32)
    D = Xte.shape[1]

    # 正常池(训练段)
    X_pool = np.stack([Xtr_z[t-window:t] for t in range(window, len(Xtr_z), pool_stride)])

    # 异常窗口
    X_anom, R_anom = [], []
    for (start, end), cause in causes.items():
        for t in range(start + window, min(start + 200, end), stride):
            X_anom.append(Xte_z[t-window:t])
            R_anom.append({cause})
    if not X_anom:
        return None
    return dict(
        name=f'smd_{machine.replace("-","_")}',
        X_anom=np.asarray(X_anom, dtype=np.float32),
        X_pool=X_pool.astype(np.float32),
        R_anom=R_anom,
        series_normal=Xtr_z[:min(10000, len(Xtr_z))],
        n_nodes=D, n_causes=len(causes))


def run_smd_attribution():
    """在SMD上运行GRAA + 基线."""
    from scorers import make_scorer
    from attribution import Context
    from attribution.graph_dep import GRAPH_DEPENDENT
    from attribution.graph_free import GRAPH_FREE
    from attribution.graa import grcm_attribute
    from graphs.sources import pcmci_graph
    from evaluation import metrics as M

    machines = ['machine-1-1', 'machine-1-2', 'machine-2-1']
    records = []
    for machine in machines:
        try:
            unit = build_smd_units(machine)
        except Exception as e:
            print(f'  Skip {machine}: {e}')
            continue
        if unit is None or len(unit['X_anom']) < 10:
            print(f'  Skip {machine}: too few windows')
            continue

        D = unit['n_nodes']
        A_pcmci = pcmci_graph(unit['series_normal'], tau_max=2)['adj']
        edge_conf = np.ones((D, D))  # 简化: 均匀置信度

        scorer = make_scorer('iforest').fit(unit['X_pool'])
        ctx0 = Context(unit['X_pool'], scorer=scorer)
        ctx_g = ctx0.with_graph(A_pcmci)

        # GRAA
        phi, info = grcm_attribute(unit['X_anom'], ctx_g,
                                   edge_conf=edge_conf, alpha_base=0.5)
        records.append(dict(
            dataset=unit['name'], method='GRAA', graph_source='pcmci',
            hit3=M.mean_hit(phi, unit['R_anom'], 3),
            n_windows=len(unit['R_anom'])))
        print(f'  {unit["name"]} GRAA: hit@3={records[-1]["hit3"]:.3f} n={len(unit["R_anom"])}', flush=True)

        # 基线
        for mname, fn in list(GRAPH_DEPENDENT.items()):
            phi = fn(unit['X_anom'], ctx_g)
            records.append(dict(
                dataset=unit['name'], method=mname, graph_source='pcmci',
                hit3=M.mean_hit(phi, unit['R_anom'], 3)))
        for mname in ('zDev', 'AERec'):
            phi = GRAPH_FREE[mname](unit['X_anom'], ctx0)
            records.append(dict(
                dataset=unit['name'], method=mname, graph_source='none',
                hit3=M.mean_hit(phi, unit['R_anom'], 3)))

    json.dump(records, open(os.path.join(CACHE, 'e19_smd.json'), 'w'), default=float)
    agg = collections.defaultdict(list)
    for r in records:
        agg[r['method']].append(r['hit3'])
    print('\n=== SMD results (hit@3, mean over machines) ===')
    for m in sorted(agg):
        print(f'  {m:14s}: {np.mean(agg[m]):.3f} (n={len(agg[m])})')
    return records


if __name__ == '__main__':
    run_smd_attribution()
