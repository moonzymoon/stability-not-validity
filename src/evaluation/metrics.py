# -*- coding: utf-8 -*-
"""双指标: ACR@K (稳定性) 与 hit@K (正确性) + RBO (v2 必改4)。

Definition (论文 §3.2):
  hit@K(A,w)   = 1{ TopK(phi_A(w)) ∩ R(w) != empty }          -- 需要金标准
  ACR@K(A,w,P) = |TopK(phi_A(w,G_m)) ∩ TopK(phi_A(w,G_ref))| / K  -- 无需金标准
双参照 (v2 必改4):
  ACR_true: G_ref = 真值图 (学术场景)
  ACR_base: G_ref = 初始估计图 G_base (部署场景)
"""
import numpy as np


def topk(phi_row: np.ndarray, K: int) -> set:
    """Top-K 变量集合; 平局按变量索引升序打破 (stable sort, 与论文声明一致)。"""
    return set(np.argsort(-phi_row, kind='stable')[:K].tolist())


def hit_at_k(phi: np.ndarray, roots, K: int = 3) -> np.ndarray:
    """phi (n,D), roots: list[set]。返回每窗口 hit (n,) in {0,1}。"""
    return np.array([1.0 if topk(phi[i], K) & set(r) else 0.0
                     for i, r in enumerate(roots)])


def mean_hit(phi: np.ndarray, roots, K: int = 3) -> float:
    return float(hit_at_k(phi, roots, K).mean())


def overlap_ratio(a: set, b: set, K: int) -> float:
    return len(a & b) / K


def acr_at_k(phi_pert: np.ndarray, phi_ref: np.ndarray, K: int = 3) -> np.ndarray:
    """逐窗口 Top-K 重叠率。phi_pert/phi_ref (n,D)。返回 (n,)。"""
    n = phi_pert.shape[0]
    return np.array([overlap_ratio(topk(phi_pert[i], K), topk(phi_ref[i], K), K)
                     for i in range(n)])


def rbo(a: np.ndarray, b: np.ndarray, p: float = 0.9) -> float:
    """Rank-Biased Overlap (多根因场景排序质量, v2 必改4)。
    单窗口级别; a,b 为两个排名向量 (变量索引按名次排列)。"""
    n = len(a)
    sa, sb = set(), set()
    out, norm = 0.0, 0.0
    for i in range(n):
        w = p ** i
        norm += w
        if i < len(a):
            sa.add(a[i])
        if i < len(b):
            sb.add(b[i])
        inter = len(sa & sb)
        d = i + 1
        out += w * inter / d if d > 0 else 0.0
    return float(out / norm) if norm > 0 else 0.0


def mean_rbo(phi: np.ndarray, roots, topn: int = 10, p: float = 0.9) -> float:
    """phi 相对金标准的平均 RBO: 参考排名 = 根因(索引升序)优先 + 非根因按索引 (论文声明的构造)。"""
    vals = []
    n, D = phi.shape
    for i, r in enumerate(roots):
        gt_rank = sorted(r) + [j for j in range(D) if j not in r]
        pred_rank = np.argsort(-phi[i])[:topn].tolist()
        vals.append(rbo(pred_rank, gt_rank[:topn], p=p))
    return float(np.mean(vals))


def quadrant_stats(points, acr_thr=0.8, hit_thr=0.4):
    """解耦象限统计: points = list of (acr, hit)。
    返回各象限占比 dict (Stable-Wrong 等)。"""
    arr = np.asarray(points, dtype=float)
    if len(arr) == 0:
        return dict(n=0, stable_wrong=0.0, stable_right=0.0,
                    unstable_right=0.0, unstable_wrong=0.0)
    n = len(arr)
    sw = np.mean((arr[:, 0] > acr_thr) & (arr[:, 1] < hit_thr))
    sr = np.mean((arr[:, 0] > acr_thr) & (arr[:, 1] >= hit_thr))
    ur = np.mean((arr[:, 0] <= acr_thr) & (arr[:, 1] >= hit_thr))
    uw = np.mean((arr[:, 0] <= acr_thr) & (arr[:, 1] < hit_thr))
    return dict(n=n, stable_wrong=float(sw), stable_right=float(sr),
                unstable_right=float(ur), unstable_wrong=float(uw))


def bootstrap_ci(vals, B=2000, seed=0, stat=np.mean):
    """均值 bootstrap 95% CI。"""
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(B, len(vals)))
    bs = stat(vals[idx], axis=1)
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
