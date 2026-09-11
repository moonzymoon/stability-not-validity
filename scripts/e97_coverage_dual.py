# -*- coding: utf-8 -*-
"""E97: 双协议覆盖-效度曲线 + fig_coverage 重画.

cross-configuration (16 折, dataset 字符串) vs strict (同种子双强度
同侧, 11 折含 TE 域折). 输出 e97_coverage_dual.json + 重画
paper/figures/fig_coverage.pdf (双曲线).
"""
import json
import os
import re

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
CACHE = os.path.join(SRC, '_cache')
FIG = os.path.join(os.path.dirname(SRC), 'paper', 'figures')

rows = json.load(open(os.path.join(CACHE, 'e3_windows.json'),
                      encoding='utf-8'))
feats = [c for c in rows[0] if len(c) > 2 and c[0] in 'gcasd'
         and c[1] == '_']
X = np.array([[r.get(f, np.nan) for f in feats] for r in rows])
y = np.array([r['label'] for r in rows])
dcol = np.array([r['dataset'] for r in rows])


def group_of(d):
    if d.startswith('scm'):
        return 'scm' + re.search(r'\d+', d).group(0)
    return 'te'


def oof(groups):
    p = np.full(len(y), np.nan)
    for gv in sorted(set(groups)):
        tr = np.array(groups) != gv
        clf = HistGradientBoostingClassifier(random_state=0)
        clf.fit(X[tr], y[tr])
        p[~tr] = clf.predict_proba(X[~tr])[:, 1]
    return p


def curve(p):
    m = ~np.isnan(p)
    pm, ym = p[m], y[m]
    order = np.argsort(-pm)
    out = []
    for cov in np.arange(1.0, 0.04, -0.05):
        k = max(1, int(round(cov * len(ym))))
        out.append(dict(coverage=float(cov),
                        accuracy=float(ym[order[:k]].mean())))
    return out


p_strict = oof([group_of(d) for d in dcol])
cur_strict = curve(p_strict)
# cross 曲线用论文原口径缓存 (e3_results.json), 保证与正文 0.67 一致
e3r = json.load(open(os.path.join(CACHE, 'e3_results.json')))
cov = e3r['coverage']
cur_cross = [dict(coverage=float(c['coverage']),
                  accuracy=float(c['accuracy'])) for c in cov]
# 若原缓存覆盖率网格稀疏, 补齐 0.05 步长点
if len(cur_cross) < 10:
    cur_cross = curve(oof(dcol))
json.dump(dict(cross=cur_cross, strict=cur_strict),
          open(os.path.join(CACHE, 'e97_coverage_dual.json'), 'w'), indent=1)
print('cross @30%:', [c for c in cur_cross if abs(c['coverage'] - 0.3)
                      < 0.01][0])
print('strict @30%:', [c for c in cur_strict if abs(c['coverage'] - 0.3)
                       < 0.01][0])

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {'blue': '#7FA4C8', 'red': '#D08AB6', 'grey': '#8E8E8E',
          'darkblue': '#3F345B'}
fig, ax = plt.subplots(figsize=(4.8, 3.6))
for cur, c, lab in (
        (cur_cross, COLORS['blue'],
         'cross-configuration folds'),
        (cur_strict, COLORS['red'],
         'strict trajectory grouping')):
    ax.plot([c['coverage'] for c in cur], [c['accuracy'] for c in cur],
            'o-', color=c, ms=5, lw=1.6, label=lab)
base = cur_strict[-1]['accuracy']
ax.axhline(base, ls=':', color=COLORS['grey'], lw=1,
           label='no selection (%.2f)' % base)
ax.set_xlabel('Coverage (fraction of windows surfaced)', fontsize=9)
ax.set_ylabel('Validity of surfaced set', fontsize=9)
ax.legend(frameon=False, fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_coverage.pdf'), dpi=300,
            bbox_inches='tight')
print('-> figures/fig_coverage.pdf (dual curves)')
