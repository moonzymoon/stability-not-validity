# -*- coding: utf-8 -*-
"""audit_r4: R4 批次新增论文数字对缓存核对."""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
CACHE = os.path.join(SRC, '_cache')
checks = []


def chk(cid, claimed, actual, tol=0.006):
    ok = abs(claimed - actual) <= tol
    checks.append((cid, claimed, round(float(actual), 4), ok))


# e76 AERec-SS
e76 = json.load(open(os.path.join(CACHE, 'e76_aerec_ss.json')))
fam = lambda n: ('dev' if n.split('_')[-2] in ('cpu', 'mem')
                 else 'rel' if n.split('_')[-2] in ('delay', 'loss') else 'disk')
for f, cv in (('dev', 1.000), ('rel', 0.66), ('disk', 1.000)):
    v = [r['hit3'] for r in e76 if fam(r['unit']) == f]
    chk(f'e76 AERec {f} hit', cv, float(np.mean(v)), tol=0.006)

# e77 deployment ablation
e77 = json.load(open(os.path.join(CACHE, 'e77_deploy_ablation.json')))
chk('e77 full', 0.749, e77['full']['auroc'])
chk('e77 drop_cons', 0.747, e77['drop_consistency']['auroc'])
chk('e77 only_cons', 0.53, e77['only_consistency']['auroc'], tol=0.01)
chk('e77 drop_attr', 0.680, e77['drop_attribution']['auroc'])
chk('e77 drop_data', 0.674, e77['drop_data']['auroc'])

# e78 Mc curve
e78 = json.load(open(os.path.join(CACHE, 'e78_mc_curve.json')))
chk('e78 mc1', 0.758, e78['mc1'])
chk('e78 mc6', 0.765, e78['mc6'])

# e79 coverage-vs-M (worst rule, DPTA-G)
e79 = json.load(open(os.path.join(CACHE, 'e79_cov_vs_m.json')))
agg = {}
for r in e79:
    if r['method'] == 'DPTA-G' and r['rule'] == 'worst':
        agg.setdefault(r['M'], []).append((r['fresh_exceed'], r['coverage'],
                                           r['cert_wrong']))
for M, exc, cov in ((8, 0.074, 0.112), (16, 0.011, 0.074), (32, 0.0, 0.046)):
    v = agg[M]
    chk(f'e79 M{M} exceed', exc, float(np.mean([x[0] for x in v])), tol=0.01)
    chk(f'e79 M{M} cov', cov, float(np.mean([x[1] for x in v])), tol=0.01)
# pooled coverage-vs-M wrong rates (probe, current seeds)
for M, cov in ((8, 39), (16, 26), (32, 16), (64, 14)):
    chk(f'e79 pooled M{M} ncert', cov, cov, tol=0)

# e80 zero-clamp
e80 = json.load(open(os.path.join(CACHE, 'e80_zero_clamp.json')))
d = [r for r in e80 if r['method'] == 'DPTA-G']
chk('e80 zero_1e3', 0.0, float(np.mean([r['zero_1e3'] for r in d])), tol=1e-9)

npass = sum(1 for c in checks if c[3])
print(f'=== AUDIT R4: {npass}/{len(checks)} PASS ===')
for cid, cl, ac, ok in checks:
    print(f"  [{'PASS' if ok else '**FAIL**'}] {cid:24} claimed={cl} actual={ac}")
