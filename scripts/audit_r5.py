# -*- coding: utf-8 -*-
"""audit_r5: 论文中 e81-e90 派生数字对缓存逐项核对 (R2 三检之第一检)."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '..', '_cache')
checks = []


def chk(cid, claimed, actual, tol=0.006):
    ok = (claimed is None and actual is None) or (
        claimed is not None and actual is not None
        and abs(claimed - actual) <= tol)
    checks.append((cid, claimed, actual, ok))


def wilson(k, n, z=1.959964):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# --- e81 案例 ---
e81 = json.load(open(os.path.join(CACHE, 'e81_case_study.json')))
chk('e81 AERec miss root', 1.0, float(e81['window_index'] >= 0 and
    not (set(e81['aerec_top3']) & set(e81['true_roots']))))
chk('e81 zDev hit root', 1.0, float(bool(set(e81['zdev_top3'])
                                        & set(e81['true_roots']))))
chk('e81 consistency 0.83', 0.83, e81['aerec_window_consistency'])
chk('e81 batch ACR 0.84', 0.84, e81['aerec_batch_acr3'])

# --- e82 十种子证书 ---
e82 = json.load(open(os.path.join(CACHE, 'e82_certcov10.json')))
d2 = e82['pooled']['DPTA-G']
q2 = {r['q']: r for r in d2}
chk('e82 q0.2 ncert 68', 68, q2[0.2]['n_certified'], tol=0)
chk('e82 q0.2 wrong 37%', 0.37, q2[0.2]['certified_wrong'], tol=0.005)
lo, hi = wilson(25, 68)
chk('e82 CI lo 26', 0.26, round(lo, 2), tol=0.005)
chk('e82 CI hi 49', 0.49, round(hi, 2), tol=0.005)
# 前5种子复现 41/19
rows5 = e82['per_seed']['DPTA-G']['0.2']
nc5 = sum(rows5[str(s)]['n_certified'] for s in range(5))
nw5 = sum(rows5[str(s)]['n_wrong'] for s in range(5))
chk('e82 seeds0-4 ncert 41', 41, nc5, tol=0)
chk('e82 seeds0-4 wrong 19', 19, nw5, tol=0)
chk('e82 q0.05 164', 164, q2[0.05]['n_certified'], tol=0)
chk('e82 q0.05 39%', 0.39, q2[0.05]['certified_wrong'], tol=0.005)
chk('e82 q0.3 49/29%', 49, q2[0.3]['n_certified'], tol=0)
pr = {r['q']: r for r in e82['pooled']['PropRank']}
chk('e82 PropRank 0@all', 0, sum(pr[q]['n_certified'] for q in pr), tol=0)

# --- e83 深度 scorer 网格 ---
e83 = json.load(open(os.path.join(CACHE, 'e83_ae_grid.json')))
import collections
agg = collections.defaultdict(list)
hit = collections.defaultdict(list)
for r in e83:
    if r['family'] == 'none':
        hit[r['method']].append(r['hit3'])
    elif r['acr3_base'] is not None:
        agg[r['method']].append(r['acr3_base'])
for m, (h3, a3) in {
        'DPTA-G': (0.53, 0.97), 'PropRank': (0.56, 0.72),
        'GraphGranger': (0.38, 0.80), 'GlobalCF': (0.44, 0.81),
        'AERec': (0.42, 0.82), 'Grad': (0.43, 0.86),
        'zDev': (0.53, 0.86), 'CondAttr': (0.46, 0.76),
        'Random': (0.27, 1.00)}.items():
    chk(f'e83 {m} hit', h3, float(np.mean(hit[m])))
    chk(f'e83 {m} acr', a3, float(np.mean(agg[m])))

# --- e84/e85 WADI ---
e84 = json.load(open(os.path.join(CACHE, 'e84_wadi.json')))
chk('e84 detection 0.80', 0.80, e84['detection_auroc'], tol=0.005)
chk('e84 d 91', 91, e84['d'], tol=0)
chk('e84 windows 272', 272, e84['n_windows'], tol=0)
chk('e84 edges 1220', 1220, e84['graph_edges'], tol=0)
e85 = json.load(open(os.path.join(CACHE, 'e85_wadi_ext.json')))
chk('e85 zeroshot 0.57', 0.57, e85['zeroshot']['auroc'], tol=0.005)
pm = e85['zeroshot']['per_method']
chk('e85 per-method min 0.36', 0.36, min(pm.values()), tol=0.005)
chk('e85 per-method max 0.71', 0.71, max(pm.values()), tol=0.005)
chk('e85 prevalence 0.28', 0.28, e85['zeroshot']['label_prev'], tol=0.005)
for m, (nc, cw) in {'zDev': (254, 0.70), 'AERec': (107, 0.72),
                    'DPTA-G': (26, 1.00), 'PropRank': (9, 1.00),
                    'Grad': (32, 0.66), 'Random': (272, 0.94)}.items():
    chk(f'e85 cert {m} n', nc, e85['cert'][m]['n_certified'], tol=0)
    chk(f'e85 cert {m} wrong', cw, e85['cert'][m]['certified_wrong'],
        tol=0.005)
chk('e85 DPTA-G hit3 0.45', 0.45, e85['methods']['DPTA-G']['hit3'])
chk('e85 DPTA-G acr3 0.89', 0.89, e85['methods']['DPTA-G']['acr3'])
chk('e85 zDev acr3 1.00', 1.00, e85['methods']['zDev']['acr3'])
chk('e85 AERec acr3 0.99', 0.99, e85['methods']['AERec']['acr3'])

# --- e87 三打分器 ---
e87 = json.load(open(os.path.join(CACHE, 'e87_wadi_scorers.json')))
chk('e87 pca GlobalCF hit 0.36', 0.36, e87['pca']['GlobalCF']['hit3'])
chk('e87 ae GlobalCF acr 0.99', 0.99, e87['ae']['GlobalCF']['acr3'])
chk('e87 ae GlobalCF sw 0.79', 0.79, e87['ae']['GlobalCF']['sw'])
chk('e87 pca zDev 1.00', 1.00, e87['pca']['zDev']['acr3'])
sa = e87['swat_ae']
chk('e87 swat_ae DPTA-G 0.62', 0.62, sa['DPTA-G']['hit3'])
chk('e87 swat_ae PropRank 0.60', 0.60, sa['PropRank']['hit3'])

# --- e88 swap 零效应 ---
e88 = json.load(open(os.path.join(CACHE, 'e88_wadi_swap.json')))
chk('e88 corr_pool 1220', 1220, e88['graph_edges']['corr_pool'], tol=0)
chk('e88 corr_contam 1232', 1232, e88['graph_edges']['corr_contam'],
    tol=0)

# --- e89 删除曲线 ---
e89 = json.load(open(os.path.join(CACHE, 'e89_deletion.json')))
chk('e89 Random 0.29', 0.29, e89['RandomDel']['drop@3'], tol=0.005)
drops = [v['drop@3'] for k, v in e89.items()
         if k not in ('RandomDel', 'Random', '_corr_drop3_hit3') and 'drop@3' in v]
chk('e89 range lo 0.33', 0.33, min(drops), tol=0.005)
chk('e89 range hi 0.50', 0.50, max(drops), tol=0.005)
chk('e89 corr 0.35', 0.35, e89['_corr_drop3_hit3'], tol=0.005)

# --- e90 R3 ---
e90 = json.load(open(os.path.join(CACHE, 'e90_r3_quant.json')))
chk('e90 trigger 1.0%', 0.010, e90['trigger_rate'], tol=0.0005)
chk('e90 wrong trig 70%', 0.70, e90['wrong_given_trigger'], tol=0.005)

# --- e86 模型类 ---
e86 = json.load(open(os.path.join(CACHE, 'e86_model_compare.json')))
for m, v in {'logistic': 0.59, 'mlp': 0.69, 'hgb': 0.81,
             'randomforest': 0.87}.items():
    chk(f'e86 {m}', v, e86[m]['lodo_mean'])

npass = sum(1 for c in checks if c[3])
print(f'=== AUDIT R5 (e81-e90): {npass}/{len(checks)} PASS ===')
for cid, cl, ac, ok in checks:
    if not ok:
        print(f"  [**FAIL**] {cid:32} claimed={cl} actual="
              f"{round(ac, 4) if isinstance(ac, float) else ac}")
print('PASS details suppressed' if npass == len(checks) else '')
