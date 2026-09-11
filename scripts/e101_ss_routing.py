# -*- coding: utf-8 -*-
"""E101: 体制路由扩展到 Sock Shop 故障族 (措施二).

路由对: zDev (偏读) vs PropRank (传输). 体制 = 故障族 (resource: cpu/mem/disk
vs network: delay/loss). 路由器 = HGB 窗口特征 -> 故障族, 按服务留出
(leave-one-service-out). 对比 always-zDev / always-PropRank / per-window
oracle. -> _cache/e101_ss_routing.json
"""
import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '..', '_cache')

rows = json.load(open(os.path.join(CACHE, 'e67_ss_windows.json'),
                      encoding='utf-8'))
feats = [c for c in rows[0] if len(c) > 2 and c[0] in 'gcasd' and c[1] == '_']

NETWORK = ('delay', 'loss')


def fam_of(ds):
    return 'network' if any(f'_{f}_' in ds for f in NETWORK) else 'resource'


def svc_of(ds):
    return ds.split('_')[1]


# 索引: (dataset, window) -> {method: label}
lab = {}
feat_by_key = {}
fam = {}
for r in rows:
    key = (r['dataset'], r['window'])
    lab.setdefault(key, {})[r['method']] = r['label']
    feat_by_key.setdefault(key, [r.get(f, np.nan) for f in feats])
    fam[key] = fam_of(r['dataset'])

keys = [k for k in lab if 'zDev' in lab[k] and 'PropRank' in lab[k]]
services = sorted(set(svc_of(k[0]) for k in keys))
print(f'windows {len(keys)} | services {services}', flush=True)

X = np.array([feat_by_key[k] for k in keys])
yfam = np.array([1 if fam[k] == 'network' else 0 for k in keys])
hz = np.array([lab[k]['zDev'] for k in keys])
hp = np.array([lab[k]['PropRank'] for k in keys])
svc = np.array([svc_of(k[0]) for k in keys])
oracle = np.maximum(hz, hp)

oof = np.full(len(keys), np.nan)
for s in services:
    tr, te = svc != s, svc == s
    clf = HistGradientBoostingClassifier(random_state=0)
    clf.fit(X[tr], yfam[tr])
    oof[te] = clf.predict_proba(X[te])[:, 1]
m = ~np.isnan(oof)
auc = roc_auc_score(yfam[m], oof[m])
print(f'LOSO family AUROC: {auc:.3f}', flush=True)

out = dict(family_auroc_loso=float(auc), n=len(keys))
for thr in (0.3, 0.5, 0.7):
    route_net = oof > thr
    sel = np.where(route_net, hp, hz)
    out[f'thr{thr}'] = dict(
        routed=float(sel[m].mean()),
        always_zdev=float(hz[m].mean()),
        always_prop=float(hp[m].mean()),
        oracle=float(oracle[m].mean()),
        frac_network=float(route_net[m].mean()))
    print(f'thr={thr}: routed={sel[m].mean():.3f} zDev={hz[m].mean():.3f} '
          f'Prop={hp[m].mean():.3f} oracle={oracle[m].mean():.3f} '
          f'(%net={route_net[m].mean():.2f})', flush=True)

# 按族分解
net_t, res_t = yfam[m] == 1, yfam[m] == 0
out['by_family'] = dict(
    network=dict(zdev=float(hz[m][net_t].mean()),
                 prop=float(hp[m][net_t].mean())),
    resource=dict(zdev=float(hz[m][res_t].mean()),
                  prop=float(hp[m][res_t].mean())))
print('network: zDev %.2f Prop %.2f | resource: zDev %.2f Prop %.2f' % (
    hz[m][net_t].mean(), hp[m][net_t].mean(),
    hz[m][res_t].mean(), hp[m][res_t].mean()), flush=True)

json.dump(out, open(os.path.join(CACHE, 'e101_ss_routing.json'), 'w'),
          indent=1)
print('-> e101_ss_routing.json')
