# -*- coding: utf-8 -*-
"""audit_v4: 任务C — 论文v4新增数字逐项对缓存核对 (23项断言表).

规则: 期望值=论文所写; 实际值=从缓存重算。2位小数容差 0.005。
FAIL 项处理: 查缓存确认 -> 改论文(记入审计报告), 不改缓存。
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import sys
import json
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
CACHE = os.path.join(SRC, '_cache')

checks = []


def chk(cid, claimed, actual, tol=0.005, note=''):
    if isinstance(claimed, (int, float)) and isinstance(actual, (int, float)):
        ok = abs(claimed - actual) <= tol
    else:
        ok = claimed == actual
    checks.append((cid, claimed, actual, ok, note))


def rng_chk(cid, lo_c, hi_c, vals, note=''):
    lo, hi = float(min(vals)), float(max(vals))
    ok = abs(lo - lo_c) <= 0.006 and abs(hi - hi_c) <= 0.006
    checks.append((cid, f'[{lo_c},{hi_c}]', f'[{lo:.3f},{hi:.3f}]', ok, note))


def mean_by(rows, **flt):
    key = lambda r: r
    sel = [r for r in rows if all(r.get(k) == v for k, v in flt.items())]
    return sel


def main():
    # ---------- e65 AttNN ----------
    e65 = json.load(open(os.path.join(CACHE, 'e65_attnn_run1.json')))
    cells, rnd = e65['cells'], e65['random_control']
    for sk, c_h, c_a in (('iforest', 0.460, 0.817), ('pca', 0.249, 0.761)):
        sub = [c for c in cells if c['scorer'] == sk]
        chk(f'1/2 AttNN {sk} hit', c_h, float(np.mean([c['hit3'] for c in sub])))
        chk(f'1/2 AttNN {sk} acr', c_a, float(np.mean([c['acr3_input'] for c in sub])))
    for sk, c_h, c_a in (('iforest', 0.263, 0.201), ('pca', 0.223, 0.202)):
        sub = [r for r in rnd if r['scorer'] == sk]
        chk(f'3 Random {sk} hit', c_h, float(np.mean([r['hit3'] for r in sub])))
        chk(f'3 Random {sk} acr', c_a, float(np.mean([r['acr3_input'] for r in sub])))
    h = [c['hit3'] for c in cells]
    rng_chk('4 AttNN seed range', 0.03, 0.81, h)
    fids = [c['train']['val_corr'] for c in cells]
    chk('5 fidelity mean', 0.22, float(np.mean(fids)), tol=0.02)
    s1 = [c for c in cells if c['seed'] == 1 and c['scorer'] == 'iforest'][0]
    chk('6 s1-iforest corr', 0.08, s1['train']['val_corr'], tol=0.01)
    chk('6 s1-iforest hit', 0.81, s1['hit3'], tol=0.01)

    # ---------- holdout table (e58 + e66_ob_holdout) ----------
    e58 = json.load(open(os.path.join(CACHE, 'e58_rcaeval_acr.json')))
    obh = json.load(open(os.path.join(CACHE, 'e66_ob_holdout.json')))
    def agg(rows, src):
        t = collections.defaultdict(lambda: ([], []))
        for r in rows:
            if r.get('method') in ('PropRank', 'DPTA-G', 'GRAA(p=0.4)') \
               and r['graph_source'] == src:
                t[r['method']][0].append(r['acr3'])
                t[r['method']][1].append(r['hit3'])
        return {m: (float(np.mean(a)), float(np.mean(h)))
                for m, (a, h) in t.items()}
    e58c, e58a = agg(e58, 'pcmci_clean'), agg(e58, 'pcmci_anom')
    obt = {}
    for r in obh:
        if r.get('method') in ('PropRank', 'DPTA-G', 'GRAA(p=0.4)'):
            obt.setdefault((r['method'], r['graph_source']), ([], []))
            obt[(r['method'], r['graph_source'])][0].append(r['acr3'])
            obt[(r['method'], r['graph_source'])][1].append(r['hit3'])
    obh_m = {k: (float(np.mean(a)), float(np.mean(h))) for k, (a, h) in obt.items()}
    obh_units = len({r['unit'] for r in obh})
    chk('8 holdout units', 17, obh_units, tol=0)
    tab11 = [('PropRank', 'pcmci_clean', .85, .52, .88, .39),
             ('PropRank', 'pcmci_anom', .85, .46, .87, .32),
             ('DPTA-G', 'pcmci_clean', .92, .80, .93, .58),
             ('DPTA-G', 'pcmci_anom', .93, .80, .93, .74),
             ('GRAA(p=0.4)', 'pcmci_clean', .92, .60, .93, .36),
             ('GRAA(p=0.4)', 'pcmci_anom', .91, .58, .95, .30)]
    for m, g, a58, h58, aob, hob in tab11:
        chk(f'7 T11 {m[:6]} {g[-5:]} in-ACR', a58, e58c[m][0] if g == 'pcmci_clean' else e58a[m][0])
        chk(f'7 T11 {m[:6]} {g[-5:]} in-hit', h58, e58c[m][1] if g == 'pcmci_clean' else e58a[m][1])
        chk(f'7 T11 {m[:6]} {g[-5:]} ho-ACR', aob, obh_m[(m, g)][0])
        chk(f'7 T11 {m[:6]} {g[-5:]} ho-hit', hob, obh_m[(m, g)][1])
    chk('9 GRAA-PropRank holdout anom hit', 0.32,
        obh_m[('PropRank', 'pcmci_anom')][1])
    chk('9 GRAA holdout anom hit', 0.30, obh_m[('GRAA(p=0.4)', 'pcmci_anom')][1])

    # ---------- e66 SS table 12 + shares ----------
    ss = json.load(open(os.path.join(CACHE, 'e66_rcaeval2.json')))
    sst = collections.defaultdict(lambda: ([], []))
    sw = collections.defaultdict(list)
    for r in ss:
        if r.get('method') in ('PropRank', 'DPTA-G', 'GRAA(p=0.4)', 'zDev'):
            key = (r['method'], r['graph_source'])
            sst[key][0].append(r['hit3'])
            if r.get('acr3') is not None:
                sst[key][1].append(r['acr3'])
                sw[r['method']].append(1.0 if (r['acr3'] > 0.8 and r['hit3'] < 0.4) else 0.0)
    tab12 = [('PropRank', 'pcmci_clean', .431, .821), ('PropRank', 'pcmci_anom', .481, .821),
             ('DPTA-G', 'pcmci_clean', .683, .889), ('DPTA-G', 'pcmci_anom', .711, .896),
             ('GRAA(p=0.4)', 'pcmci_clean', .517, .873), ('GRAA(p=0.4)', 'pcmci_anom', .534, .889),
             ('zDev', 'pcmci_clean', .682, None), ('zDev', 'pcmci_anom', .682, None)]
    for m, g, ch, ca in tab12:
        hs = sst[(m, g)][0]
        chk(f'10 T12 {m[:6]} {g[-5:]} hit', ch, float(np.mean(hs)))
        if ca is not None:
            chk(f'10 T12 {m[:6]} {g[-5:]} acr', ca, float(np.mean(sst[(m, g)][1])))
    for m, cs in (('PropRank', 31.9), ('DPTA-G', 21.5), ('GRAA(p=0.4)', 31.9)):
        chk(f'11 SW {m}', cs, float(np.mean(sw[m])) * 100, tol=0.15)

    # ---------- e73 significance ----------
    e73 = json.load(open(os.path.join(CACHE, 'e73_signif_ss.json')))
    for r in e73:
        if r['graph'] == 'pcmci_anom' and r['comparison'] == 'GRAA-PropRank':
            chk('12 anom diff', 0.052, r['mean_diff'])
            chk('12 anom p<0.001', 1, int(r['p'] < 0.001), tol=0)
        if r['graph'] == 'pcmci_clean' and r['comparison'] == 'GRAA-PropRank':
            chk('12 clean diff', 0.087, r['mean_diff'])
            chk('12 clean p=0.011', 0.011, r['p'], tol=0.002)

    # ---------- e67 predictor ----------
    e67 = json.load(open(os.path.join(CACHE, 'e67_results.json')))['results']
    lodo = [r['auroc'] for r in e67 if r['protocol'] == 'SS_LODO_service']
    rng_chk('15 LODO range', 0.81, 0.86, lodo)
    chk('15 LODO mean', 0.84, float(np.mean(lodo)), tol=0.01)
    chk('15 pooled', 0.85, [r['auroc'] for r in e67
                            if r['protocol'] == 'SS_pooled_GroupKFold5'][0], tol=0.005)
    zs = {r['protocol']: r['auroc'] for r in e67 if 'zero-shot' in r['protocol']}
    chk('16 zero SS->OBh', 0.85, zs['zero-shot SS->OBh'], tol=0.005)
    chk('16 zero OBh->SS', 0.76, zs['zero-shot OBh->SS'], tol=0.005)
    rc = [r['auroc'] for r in e67 if r['protocol'].startswith('recal')][0]
    chk('16 recal', 0.75, rc, tol=0.005)

    # ---------- e68 ----------
    e68 = json.load(open(os.path.join(CACHE, 'e68_attnn_ext.json')))
    t68 = collections.defaultdict(list)
    for r in e68:
        t68[(r['testbed'], r['scorer'])].append((r['hit3'], r['acr3_input']))
    exp68 = {('ss', 'iforest'): (.457, .960), ('ss', 'pca'): (.542, .965),
             ('ob_holdout', 'iforest'): (.139, .966), ('ob_holdout', 'pca'): (.294, .973)}
    for k, (ch, ca) in exp68.items():
        v = t68[k]
        chk(f'17 {k[0]}/{k[1]} hit', ch, float(np.mean([x[0] for x in v])))
        chk(f'17 {k[0]}/{k[1]} acr', ca, float(np.mean([x[1] for x in v])))
    gmeans = [float(np.mean([x[1] for x in v])) for v in t68.values()]
    rng_chk('17 acr group-mean range 0.96-0.973', 0.96, 0.973, gmeans)

    # ---------- e69/e72 AERCA ----------
    e69 = json.load(open(os.path.join(CACHE, 'e69_aerca_ss.json')))
    chk('18 AERCA-SS mean', 0.073, float(np.mean([v['hit3'] for v in e69.values()])), tol=0.002)
    zd = [r['hit3'] for r in ss if r['method'] == 'zDev' and r['graph_source'] == 'pcmci_clean']
    chk('18 zDev SS', 0.68, float(np.mean(zd)), tol=0.005)
    e72 = json.load(open(os.path.join(CACHE, 'e72_aerca_te.json')))
    chk('19 AERCA-TE mean', 0.013, float(np.mean([v['hit3'] for v in e72.values()])), tol=0.002)
    e24 = json.load(open(os.path.join(CACHE, 'e24_graa_te.json')))
    grp = collections.defaultdict(list)
    for r in e24:
        if r['method'] in ('PropRank', 'DPTA-G', 'GRAA(p=0.4)'):
            grp[(r['method'], r['graph_source'])].append(r['hit3'])
    te_means = [float(np.mean(v)) for v in grp.values() if len(v) >= 5]
    rng_chk('19 TE graph family MEANS 0.115-0.246', 0.115, 0.246, te_means)

    # ---------- e70 adversarial ----------
    e70 = json.load(open(os.path.join(CACHE, 'e70_adversarial_ss.json')))
    a70 = collections.defaultdict(list)
    for r in e70:
        for k in ('random_q0.1', 'random_q0.2', 'adversarial_B2'):
            a70[(r['method'], k)].append(r[k])
    for m, k, cv in (('PropRank', 'random_q0.1', .744), ('PropRank', 'adversarial_B2', .350),
                     ('DPTA-G', 'random_q0.1', .881), ('DPTA-G', 'adversarial_B2', .644)):
        chk(f'20 {m} {k}', cv, float(np.mean(a70[(m, k)])))

    # ---------- e71/e62 certificates ----------
    e71 = json.load(open(os.path.join(CACHE, 'e71_cert_ss.json')))
    for m, cov50, fl50, wrong in (('PropRank', 43.4, .454, .536), ('DPTA-G', 56.9, .302, .280)):
        sub = [r for r in e71 if r['method'] == m]
        cert = [r for r in sub if r['cert_emp50']]
        chk(f'21 {m} emp50 cov', cov50, float(np.mean([r["cert_emp50"] for r in sub])) * 100, tol=0.15)
        chk(f'21 {m} emp50 flip', fl50, float(np.mean([r['flip_eval'] for r in cert])))
        chk(f'21 {m} emp50 wrong', wrong, float(1 - np.mean([r['label'] for r in cert])))
    wcov = [np.mean([r['cert_worst'] for r in e71 if r['method'] == m])
            for m in ('PropRank', 'DPTA-G')]
    rng_chk('21 worst cov 0.35-3.47%', 0.347, 3.472, [x * 100 for x in wcov], )
    e62 = json.load(open(os.path.join(CACHE, 'e62_te_cert.json')))
    sub62 = [r for r in e62 if r['method'] == 'DPTA-G' and r['cert_worst']]
    chk('22 TE worst cov', 11.5, float(np.mean([r['cert_worst'] for r in e62
                                                if r['method'] == 'DPTA-G'])) * 100, tol=0.15)
    chk('22 TE cert wrong', 97, float(1 - np.mean([r['label'] for r in sub62])) * 100, tol=0.5)

    # ---------- report ----------
    npass = sum(1 for c in checks if c[3])
    print(f'=== AUDIT: {npass}/{len(checks)} PASS ===')
    for cid, cl, ac, ok, note in checks:
        flag = 'PASS' if ok else '**FAIL**'
        print(f'  [{flag}] {cid:34} claimed={cl} actual={ac}' + (f' ({note})' if note else ''))
    json.dump([dict(id=c[0], claimed=str(c[1]), actual=str(c[2]), ok=bool(c[3]))
               for c in checks],
              open(os.path.join(CACHE, 'audit_v4.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
