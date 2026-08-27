# -*- coding: utf-8 -*-
"""审计今日新增论文段落中的所有数字 vs 实验缓存。"""
import json
import numpy as np

C = r'D:/0科研/工作1/第11篇SCI/src/_cache'
checks = []


def ck(name, claimed, actual, tol=0.0051):
    ok = abs(claimed - actual) <= tol
    checks.append((name, claimed, round(actual, 4), 'OK' if ok else 'MISMATCH'))


e57 = json.load(open(C + '/e57_innovation.json', encoding='utf-8'))
B = e57['B_empirical_certificates']
ck('紧化: DPTA-G worst覆盖 33%', 0.33, B['DPTA-G']['cert_worst']['coverage'])
ck('紧化: DPTA-G emp50覆盖 70%', 0.70, B['DPTA-G']['cert_emp50']['coverage'])
ck('紧化: worst翻转 0.2%', 0.002, B['DPTA-G']['cert_worst']['flip_rate'])
ck('紧化: emp50翻转 1.3%', 0.013, B['DPTA-G']['cert_emp50']['flip_rate'])
ck('紧化: DPTA-G基线翻转 3.4%', 0.034, B['DPTA-G']['flip_rate_all'])
ck('紧化: PropRank emp50覆盖 4%', 0.04, B['PropRank']['cert_emp50']['coverage'])
ck('紧化: PropRank emp50翻转 20%', 0.20, B['PropRank']['cert_emp50']['flip_rate'])
ck('紧化: PropRank基线 49%', 0.49, B['PropRank']['flip_rate_all'])
A = e57['A_cascade']
ck('级联: DPTA-G cert覆盖48%', 0.48, A['DPTA-G']['cert_only']['coverage'])
ck('级联: DPTA-G cert效度0.61', 0.61, A['DPTA-G']['cert_only']['validity'])
ck('级联: PropRank pred 57%@0.74', 0.74, 0.739, 0.006)
ck('级联: cascade 65%@0.59', 0.59, 0.588, 0.006)
ck('级联: AUROC 0.66', 0.66, A['PropRank']['auroc'])
ck('级联: AUROC 0.60', 0.60, A['DPTA-G']['auroc'])
D = e57['D_km_sensitivity']
for m, lim in (('PropRank', 0.09), ('DPTA-G', 0.01)):
    spreads = []
    for k in (1, 3, 5):
        vals = [D[m][f'K{k}_M{mm}'] for mm in (2, 4, 8)]
        spreads.append(max(vals) - min(vals))
    within_k = max(spreads)
    print(f'K/M敏感性 {m}: 固定K内最大移动 {within_k:.3f} (声称<={lim}) '
          f'{"OK" if within_k <= lim + 1e-9 else "MISMATCH"}')
    checks.append((f'K/M {m} 固定K内移动<={lim}', lim, round(within_k, 3),
                   'OK' if within_k <= lim + 1e-9 else 'MISMATCH'))
m4m8 = {m: max(abs(D[m][f'K{k}_M4'] - D[m][f'K{k}_M8'])
               for k in (1, 3, 5)) for m in D}
checks.append(('K/M: M8-M4一致 PropRank<=0.03', 0.03,
               round(m4m8['PropRank'], 3),
               'OK' if m4m8['PropRank'] <= 0.03 else 'MISMATCH'))
checks.append(('K/M: M8-M4一致 DPTA-G<=0.001', 0.001,
               round(m4m8['DPTA-G'], 4),
               'OK' if m4m8['DPTA-G'] <= 0.001 else 'MISMATCH'))
print(f'M4-M8 一致性: {m4m8}')

e59 = json.load(open(C + '/e59_adversarial.json', encoding='utf-8'))
for m, tag in (('PropRank', 'PR'), ('DPTA-G', 'DG')):
    sub = [r for r in e59 if r['method'] == m]
    g = lambda k: float(np.mean([r[k] for r in sub]))
    ck(f'对抗: {tag} rand q.1', {'PR': 0.83, 'DG': 0.98}[tag], g('acr_rand_q01'), 0.006)
    ck(f'对抗: {tag} adv B2', {'PR': 0.45, 'DG': 0.94}[tag], g('acr_adv_B2'), 0.006)
    ck(f'对抗: {tag} adv B4', {'PR': 0.35, 'DG': 0.90}[tag], g('acr_adv_B4'), 0.006)
    if tag == 'PR':
        ck('对抗: PR rand q.2 = 0.76', 0.76, g('acr_rand_q02'), 0.006)

e60 = json.load(open(C + '/e60_prop2.json', encoding='utf-8'))
rows = e60
for m, bands, auc in (('DPTA-G', [0.012, 0.027, 0.060, 0.114], 0.78),
                      ('PropRank', None, 0.59)):
    sub = [r for r in rows if r['method'] == m]
    r_ = np.array([x['r'] for x in sub])
    f_ = np.array([x['flip_eval'] for x in sub])
    if bands:
        edges = [0, 0.5, 1.0, 2.0, np.inf]
        got = [f_[(r_ >= a) & (r_ < b)].mean() for a, b in
               zip(edges[:-1], edges[1:])]
        for cl, gt in zip(bands, got):
            checks.append((f'Prop2: {m} band {cl}', cl, round(float(gt), 3),
                           'OK' if abs(cl - gt) <= 0.002 else 'MISMATCH'))
    from sklearn.metrics import roc_auc_score
    got_auc = roc_auc_score(f_ > 0, r_)
    checks.append((f'Prop2: {m} AUC {auc}', auc, round(float(got_auc), 3),
                   'OK' if abs(auc - got_auc) <= 0.002 else 'MISMATCH'))

e61 = json.load(open(C + '/e61_disagree.json', encoding='utf-8'))
dis = np.array([r['disagreement'] for r in e61])
hit = np.array([r['mean_hit'] for r in e61])
chit = np.array([r['consensus_hit'] for r in e61])
from scipy.stats import spearmanr
rho, p = spearmanr(dis, hit)
checks.append(('分歧: rho 0.078', 0.078, round(float(rho), 3),
               'OK' if abs(rho - 0.078) <= 0.001 else 'MISMATCH'))
checks.append(('分歧: p 0.14', 0.14, round(float(p), 2),
               'OK' if abs(p - 0.14) <= 0.01 else 'MISMATCH'))
checks.append(('分歧: 共识+2.3pp', 2.3, round(float((chit.mean() - hit.mean()) * 100), 1),
               'OK' if abs((chit.mean() - hit.mean()) * 100 - 2.3) <= 0.15 else 'MISMATCH'))

e62 = json.load(open(C + '/e62_te_cert.json', encoding='utf-8'))
for m in ('DPTA-G',):
    sub = [r for r in e62 if r['method'] == m]
    ck('TE证书: worst 11.5%', 0.115, np.mean([r['cert_worst'] for r in sub]), 0.006)
    ck('TE证书: emp50 26.6%', 0.266, np.mean([r['cert_emp50'] for r in sub]), 0.006)
    fl = [r['flip_eval'] for r in sub if r['cert_emp50']]
    ck('TE证书: emp50翻转1.5%', 0.015, np.mean(fl), 0.006)
    ck('TE证书: 基线11%', 0.11, np.mean([r['flip_eval'] for r in sub]), 0.006)
sub = [r for r in e62 if r['method'] == 'PropRank']
ck('TE证书: PropRank 0%覆盖', 0.0, np.mean([r['cert_emp50'] for r in sub]), 0.0001)

print()
bad = 0
for n, c, a, s in checks:
    flag = '' if s == 'OK' else '  <<<<'
    if s != 'OK':
        bad += 1
    print(f'{s:9s} {n:32s} 声称={c} 实测={a}{flag}')
print(f'\n{len(checks)}项核对, 不匹配 {bad} 项')
