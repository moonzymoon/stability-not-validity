# -*- coding: utf-8 -*-
"""E38: 异常类型感知读取器选择器 — 能否免标签检测体制并路由?

训练: HGB 分类器判 deviational(0)/relational(1), 特征 = PropRank 行的
可观测特征 (selector 监视部署中的图消费者).
路由: P(rel)>thr -> AERec, 否则 PropRank.
评估 (LOO-seed): selector vs 恒PropRank vs 恒AERec vs oracle(逐窗取优),
阈值 0.3/0.5/0.7. 输出: _cache/e38_selector.json
"""
import json
import collections

import numpy as np

CACHE = r'D:\0科研\工作1\第11篇SCI\src\_cache'

dev = json.load(open(f'{CACHE}\\e3_windows.json', encoding='utf-8'))
rel = json.load(open(f'{CACHE}\\e9_relation_windows.json', encoding='utf-8'))


def index(rows, method, source=None):
    out = {}
    for r in rows:
        if r['method'] != method:
            continue
        if source is not None and r.get('graph_source') != source:
            continue
        key = (r['dataset'], r.get('scorer', ''), r['window'])
        out[key] = r
    return out


# 用 PropRank/true 图行做特征源; AERec 行提供重构读取器的 hit
feats_dev = index(dev, 'PropRank', source='true')
feats_rel = index(rel, 'PropRank', source='true')
ae_dev = index(dev, 'AERec')
ae_rel = index(rel, 'AERec')

feat_cols = [c for c in next(iter(feats_dev.values()))
             if c[0] in 'gcasd' and c[1] == '_']


def build(feats, ae, strand):
    X, y, hp, ha, ds = [], [], [], [], []
    for k, r in feats.items():
        if r.get('graph_source', 'true') != 'true':
            continue
        a = ae.get(k)
        if a is None:
            continue
        X.append([r.get(c, np.nan) for c in feat_cols])
        y.append(strand)
        hp.append(r['label'])
        ha.append(a['label'])
        ds.append(r['dataset'])
    return np.array(X, float), np.array(y), np.array(hp), np.array(ha), ds


Xd, yd, hd_prop_dev, hd_ae_dev, dsd = build(feats_dev, ae_dev, 0)
Xr, yr, hr_prop_rel, hr_ae_rel, dsr = build(feats_rel, ae_rel, 1)
print(f'dev rows {len(yd)} (AERec rate {hd_ae_dev.mean():.3f}, '
      f'PropRank {hd_prop_dev.mean():.3f})')
print(f'rel rows {len(yr)} (AERec {hr_ae_rel.mean():.3f}, '
      f'PropRank {hr_prop_rel.mean():.3f})')

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# LOO-seed: 按数据集前缀 (scm0..4 / rel_scm0..4) 分组
def seeds_of(ds_list):
    return np.array([d.split('_')[0].replace('rel', '').replace('scm', '')
                     for d in ds_list])


sd = seeds_of(dsd)
sr = seeds_of(dsr)

res = {'thresholds': {}, 'summary': {}}
pooled_pred = np.zeros(len(yd) + len(yr))
pooled_true = np.concatenate([yd, yr])
pooled_hp = np.concatenate([hd_prop_dev, hr_prop_rel])
pooled_ha = np.concatenate([hd_ae_dev, hr_ae_rel])

for s in sorted(set(sd) | set(sr)):
    trd, ted = sd != s, sd == s
    trr, ter = sr != s, sr == s
    Xtr = np.vstack([Xd[trd], Xr[trr]])
    ytr = np.concatenate([yd[trd], yr[trr]])
    clf = HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                         learning_rate=0.08, random_state=0)
    clf.fit(Xtr, ytr)
    Xte = np.vstack([Xd[ted], Xr[ter]])
    p = clf.predict_proba(Xte)[:, 1]
    pooled_pred[np.concatenate([ted, ter])] = p

auc = roc_auc_score(pooled_true, pooled_pred)
print(f'\nLOO-seed strand AUC: {auc:.3f}')

for thr in (0.3, 0.5, 0.7):
    route_ae = pooled_pred > thr
    sel = np.where(route_ae, pooled_ha, pooled_hp)
    res['thresholds'][str(thr)] = dict(
        selector=float(sel.mean()),
        always_prop=float(pooled_hp.mean()),
        always_ae=float(pooled_ha.mean()),
        oracle=float(np.maximum(pooled_hp, pooled_ha).mean()),
        frac_ae=float(route_ae.mean()))
    print(f'thr={thr}: selector={sel.mean():.3f} '
          f'prop={pooled_hp.mean():.3f} ae={pooled_ha.mean():.3f} '
          f'oracle={np.maximum(pooled_hp, pooled_ha).mean():.3f} '
          f'(%routed-to-AE={route_ae.mean():.2f})')

# 分 strand 的路由行为 (thr=0.5)
t = 0.5
sel_d = np.where(pooled_pred[:len(yd)] > t, hd_ae_dev, hd_prop_dev)
sel_r = np.where(pooled_pred[len(yd):] > t, hr_ae_rel, hr_prop_rel)
res['summary'] = dict(
    auc=float(auc),
    deviational=dict(selector=float(sel_d.mean()),
                     prop=float(hd_prop_dev.mean()),
                     ae=float(hd_ae_dev.mean())),
    relational=dict(selector=float(sel_r.mean()),
                    prop=float(hr_prop_rel.mean()),
                    ae=float(hr_ae_rel.mean())))
print(f'\ndeviational: selector {sel_d.mean():.3f} vs prop '
      f'{hd_prop_dev.mean():.3f} (drag={hd_prop_dev.mean()-sel_d.mean():+.3f})')
print(f'relational:  selector {sel_r.mean():.3f} vs prop '
      f'{hr_prop_rel.mean():.3f} (gain={sel_r.mean()-hr_prop_rel.mean():+.3f})')

json.dump(res, open(f'{CACHE}\\e38_selector.json', 'w'), indent=1)
print('\nsaved e38_selector.json')
