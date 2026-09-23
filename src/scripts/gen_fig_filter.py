# -*- coding: utf-8 -*-
"""生成 fig_filter.pdf: 完全稳定窗过滤器的折内归一 risk-coverage 曲线.
数据: _cache/e106b_certfilter.json (e106c 折内协议) — 不改任何数字."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
CACHE = os.path.join(SRC, '_cache')
OUT = os.path.join(SRC, '..', 'paper', 'figures', 'fig_filter.pdf')

j = json.load(open(os.path.join(CACHE, 'e106b_certfilter.json'),
                   encoding='utf-8'))
cur = j['risk_coverage_foldnorm']
keep = [c['keep_frac'] for c in cur]
wrong = [c['wrong_rate'] for c in cur]
base = j['certified_wrong_rate_pooled']

plt.rcParams.update({'font.size': 8.5, 'font.family': 'DejaVu Sans'})
fig, ax = plt.subplots(figsize=(3.35, 2.35))
ax.plot([k * 100 for k in keep], [w * 100 for w in wrong], 'o-',
        color='#1a5276', lw=1.4, ms=3.6, label='keep most trusted')
ax.axhline(base * 100, ls='--', color='#922b21', lw=1.0,
           label='no filter (%.1f%% wrong)' % (base * 100))
ax.annotate('48.4%% at 20%%' , xy=(20, 48.4), xytext=(34, 55),
            fontsize=7.5, color='#1a5276',
            arrowprops=dict(arrowstyle='->', color='#1a5276', lw=0.7))
ax.set_xlabel('retained fraction of perfectly stable windows (%)')
ax.set_ylabel('residual wrong rate (%)')
ax.set_xlim(15, 103)
ax.set_ylim(40, 68)
ax.legend(fontsize=7, frameon=False, loc='upper right')
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig(OUT)
print('saved', OUT, '| keep20 wrong =', wrong[-1], '| base =', base)
