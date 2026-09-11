# -*- coding: utf-8 -*-
"""E84: WADI 测试床 (标准套件第二成员, 变量级攻击真值).

数据: WADI.A2_19 Nov 2019 (A2 标准版, Attack LABLE 列, -1=攻击);
攻击点->列 映射来自 A1 版 attack_description.xlsx (15 次攻击),
别名披露: 描述中的 2LIT002 在 A2 列名中为 2_LT_002_PV (同一物位变送器);
攻击7 为两步 (1_AIT_002 + 2_MV_003), 取并集根因。

协议 (对齐 SWaT 条目): 窗口30/步长30(1s), 正常池 2000 窗 (正常段
第2天起采样), 池统计 z-score, iforest 打分器, corr 阈值图 (|r|>0.35,
池序列 lag-0) 作为部署估计图; 九族 + Random;
ACR@3: 图依赖族 del/add/rew 0.2 混合 M=8, 图无关族 noise 0.2(MAD) M=8;
输出: 每方法 hit@1/3/5, ACR@3, 窗口级稳定-错误份额, 检测AUROC
-> _cache/e84_wadi.json
"""
import os
import sys
import json

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from scorers import make_scorer, detection_auroc
from attribution import Context
from attribution.graph_dep import GRAPH_DEPENDENT
from attribution.graph_free import GRAPH_FREE
from evaluation import metrics as M
from scripts.e1_anchor import perturb_graph, stable_seed

CACHE = os.path.join(SRC, '_cache')
WADI = (r'D:\0科研\工作1\第2篇SCI\Contrastive_TopK_MIL\datasets\WADI'
        r'\WADI.A2_19 Nov 2019')

# 段->根因映射。A2 标签文件含 14 段; 与 attack_description.xlsx 的
# 15 次攻击按 时间+时长 对齐验证 (描述中的第4次在 A2 标签中缺失;
# seg8 = 描述的"随机化攻击", 根因歧义, 保守剔除并披露):
ATTACK_ROOTS = [
    ['1_MV_001_STATUS'],                                    # seg0=攻击1
    ['1_FIT_001_PV'],                                       # seg1=攻击2
    ['2_LT_002_PV'],                                        # seg2=攻击3(别名)
    ['2_MCV_101_CO', '2_MCV_201_CO', '2_MCV_301_CO',
     '2_MCV_401_CO', '2_MCV_501_CO', '2_MCV_601_CO'],       # seg3=攻击5
    ['2_MCV_101_CO', '2_MCV_201_CO'],                       # seg4=攻击6
    ['1_AIT_002_PV', '2_MV_003_STATUS'],                    # seg5=攻击7
    ['2_MCV_007_CO'],                                       # seg6=攻击8
    ['2_PIC_003_CO', '2_PIC_003_PV', '2_PIC_003_SP'],       # seg7=攻击13
    None,                                                   # seg8 随机化, 剔除
    ['1_MV_001_STATUS'],                                    # seg9=攻击10
    ['2_MCV_007_CO'],                                       # seg10=攻击11
    ['2_MCV_007_CO'],                                       # seg11=攻击12
    ['1_P_001_STATUS', '1_P_003_STATUS'],                   # seg12=攻击14
    ['2_LT_002_PV'],                                        # seg13=攻击15
]

WIN, STRIDE, N_POOL, CORR_TH = 30, 30, 2000, 0.35
CAP_PER_SEG = 60


def load_attack():
    df = pd.read_csv(os.path.join(WADI, 'WADI_attackdataLABLE.csv'),
                     skiprows=[0], low_memory=False)
    df.columns = [str(c).strip().strip('"').strip() for c in df.columns]
    lab_col = [c for c in df.columns if 'Attack LABLE' in c][0]
    sigs = [c for c in df.columns[3:] if c != lab_col]
    lab = df[lab_col].astype(float).values
    X = df[sigs].astype(np.float32).values
    return X, lab, sigs


def load_pool_rows(sigs):
    """按列名对齐读正常文件 (chunked, 每块抽样), 返回 (n, len(sigs))."""
    path = os.path.join(WADI, 'WADI_14days_new.csv')
    have = set(sigs)
    rows = []
    for i, chunk in enumerate(pd.read_csv(path,
                                          chunksize=100000,
                                          low_memory=False)):
        chunk.columns = [str(c).strip().strip('"').strip()
                         for c in chunk.columns]
        sub = chunk[[c for c in sigs if c in chunk.columns]]
        if len(sub.columns) != len(sigs):
            missing = have - set(chunk.columns)
            if missing:
                for m in missing:
                    sub[m] = np.nan
            sub = sub[sigs]
        step = max(1, len(sub) // 40000)
        rows.append(sub.iloc[::step].astype(np.float32).values)
    return np.vstack(rows)


def segments_from_label(lab):
    at = lab == -1.0
    segs, s = [], None
    for i, v in enumerate(at):
        if v and s is None:
            s = i
        elif not v and s is not None:
            segs.append((s, i))
            s = None
    if s is not None:
        segs.append((s, len(at)))
    return segs


def main():
    Xa, lab, sigs = load_attack()
    print('attack file:', Xa.shape, '| attack rows:', int((lab == -1).sum()),
          flush=True)

    # 全攻击文件上的 NaN/常数列 (覆盖正常+攻击期)
    nan_mask = np.isnan(Xa).mean(0) > 0.99
    with np.errstate(invalid='ignore'):
        const_mask = np.nanmax(Xa, 0) == np.nanmin(Xa, 0)
    keep = ~(nan_mask | const_mask)
    sigs = [s for s, k in zip(sigs, keep) if k]
    Xa = Xa[:, keep]
    print('retained signals:', len(sigs), flush=True)

    # 正常池 (按列名对齐)
    Xp = load_pool_rows(sigs)
    with np.errstate(invalid='ignore'):
        std_p = np.nanstd(Xp, 0)
        nan_p = np.isnan(Xp).mean(0) > 0.99
    keep2 = (std_p > 1e-9) & (~nan_p)
    sigs = [s for s, k in zip(sigs, keep2) if k]
    Xa, Xp = Xa[:, keep2], Xp[:, keep2]
    col_idx = {s: i for i, s in enumerate(sigs)}
    d = len(sigs)
    print('final d =', d, flush=True)

    mu, sd = np.nanmean(Xp, 0), np.nanstd(Xp, 0)
    sd[sd == 0] = 1.0
    Xa = np.clip((Xa - mu) / sd, -8, 8)
    Xp = np.clip((Xp - mu) / sd, -8, 8)
    Xa = np.nan_to_num(Xa)
    Xp = np.nan_to_num(Xp)

    # 池窗口
    n_pool_win = N_POOL
    starts = np.linspace(0, len(Xp) - WIN - 1, n_pool_win + 2).astype(int)[1:-1]
    X_pool = np.stack([Xp[s:s + WIN] for s in starts]).astype(np.float32)

    # 攻击窗口
    segs = segments_from_label(lab)
    print('attack segments:', len(segs), flush=True)
    X_anom, R_anom, seg_ids = [], [], []
    for k, (a, b) in enumerate(segs):
        roots = ATTACK_ROOTS[k] if k < len(ATTACK_ROOTS) else None
        if roots is None:
            continue
        ridx = sorted({col_idx[c] for c in roots if c in col_idx})
        if not ridx:
            continue
        n_win = max(0, (b - a - WIN) // STRIDE + 1)
        take = min(n_win, CAP_PER_SEG)
        step = max(1, n_win // take) if n_win > take else 1
        cnt = 0
        for s in range(a, b - WIN + 1, STRIDE * step):
            if cnt >= take:
                break
            X_anom.append(Xa[s:s + WIN])
            R_anom.append(set(ridx))
            seg_ids.append(k)
            cnt += 1
    X_anom = np.stack(X_anom).astype(np.float32)
    print('anomalous windows:', len(X_anom), 'over', len(segs), 'segments',
          flush=True)

    # 打分器 + 检测健全性
    scorer = make_scorer('iforest').fit(X_pool)
    auroc = detection_auroc(scorer, X_pool, X_anom)
    print('detection AUROC (pool vs attack windows): %.3f' % auroc, flush=True)

    ctx0 = Context(X_pool, scorer=scorer)

    # corr 阈值图 (部署估计图)
    flat = Xp[starts[0]:starts[-1] + WIN]
    C = np.corrcoef(flat.T)
    A = (np.abs(C) > CORR_TH).astype(float)
    np.fill_diagonal(A, 0)
    print('corr graph: %d edges, density %.3f' % (A.sum(), A.mean()),
          flush=True)

    out = dict(d=d, n_windows=len(X_anom), n_segments=len(segs),
               detection_auroc=float(auroc), graph_edges=int(A.sum()),
               methods={}, roots_size=[len(r) for r in R_anom])

    mad = ctx0.pool_stats['mad']
    for mname, fn in list(GRAPH_DEPENDENT.items()):
        phi0 = fn(X_anom, ctx0.with_graph(A))
        acrs, cons_w = [], []
        for mi in range(8):
            rng = np.random.default_rng(
                stable_seed('e84', mname, mi))
            fam = ['del', 'add', 'rew'][mi % 3]
            A_m = perturb_graph(A, fam, 0.2, rng)
            phi_m = fn(X_anom, ctx0.with_graph(A_m))
            acrs.append(M.acr_at_k(phi_m, phi0, 3).mean())
            cons_w.append(M.acr_at_k(phi_m, phi0, 3))
        h3w = M.hit_at_k(phi0, R_anom, 3)
        cons = np.stack(cons_w).mean(0)
        out['methods'][mname] = dict(
            hit1=float(M.mean_hit(phi0, R_anom, 1)),
            hit3=float(M.mean_hit(phi0, R_anom, 3)),
            hit5=float(M.mean_hit(phi0, R_anom, 5)),
            acr3=float(np.mean(acrs)),
            stable_wrong=float(((cons > 0.8) & (h3w == 0)).mean()),
            graph=True)
        print(mname, out['methods'][mname], flush=True)

    for mname, fn in GRAPH_FREE.items():
        if mname == 'Grad':
            # e1 约定: Grad 跑在可微打分器上 -> 用 PCA 的 torch 版,
            # 其余方法保持 isolation forest (与 SCM 网格的 pca-cell 同型)
            from scorers import TorchPCAScorer, make_scorer as _mk
            pca = _mk('pca').fit(X_pool)
            ctx_g = Context(X_pool, scorer=make_scorer('iforest')
                            .fit(X_pool))
            ctx_g.torch_scorer = TorchPCAScorer(pca)
            ctx_use = ctx_g
        else:
            ctx_use = ctx0
        phi0 = fn(X_anom, ctx_use)
        acrs, cons_w = [], []
        for mi in range(8):
            rng = np.random.default_rng(stable_seed('e84', mname, mi))
            X_m = (X_anom + rng.normal(0, 1, X_anom.shape).astype(np.float32)
                   * (0.2 * mad)[None, None, :]).astype(np.float32)
            phi_m = fn(X_m, ctx_use)
            acrs.append(M.acr_at_k(phi_m, phi0, 3).mean())
            cons_w.append(M.acr_at_k(phi_m, phi0, 3))
        h3w = M.hit_at_k(phi0, R_anom, 3)
        cons = np.stack(cons_w).mean(0)
        out['methods'][mname] = dict(
            hit1=float(M.mean_hit(phi0, R_anom, 1)),
            hit3=float(M.mean_hit(phi0, R_anom, 3)),
            hit5=float(M.mean_hit(phi0, R_anom, 5)),
            acr3=float(np.mean(acrs)),
            stable_wrong=float(((cons > 0.8) & (h3w == 0)).mean()),
            graph=False)
        print(mname, out['methods'][mname], flush=True)

    path = os.path.join(CACHE, 'e84_wadi.json')
    json.dump(out, open(path, 'w'), indent=1)
    print('->', path)


if __name__ == '__main__':
    main()
