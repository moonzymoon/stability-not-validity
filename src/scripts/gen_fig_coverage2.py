# -*- coding: utf-8 -*-
"""fig_coverage 升级为双面板: (a)预测器coverage-accuracy (b)过滤器risk-coverage.
(b)数据: e106b_certfilter.json (e106c 折内). 不改任何数字."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
CACHE = os.path.join(SRC, '_cache')
OUT = os.path.join(SRC, '..', 'paper', 'figures', 'fig_coverage.pdf')

plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans'})
fig, (axa, axb) = plt.subplots(2, 1, figsize=(3.35, 4.5))

# (a) 原预测器曲线 (数据点与原 fig_coverage 一致: 0.34@100%, 0.57@50, 0.67@30, 0.40 strict@30, 0.35 strict base)
cov = [100, 50, 30]
val = [0.34, 0.57, 0.67]
axa.plot(cov, val, 'o-', color='#1a5276', lw=1.4, ms=3.6,
         label='cross-configuration')
axa.plot([30], [0.40], 's', color='#922b21', ms=3.6,
         label='strict grouping')
axa.axhline(0.35, ls=':', color='#7f8c8d', lw=0.9)
axa.annotate('no-skill 0.35', xy=(70, 0.35), xytext=(58, 0.29),
             fontsize=6.5, color='#7f8c8d')
axa.set_xlim(24, 104)
axa.set_ylim(0.25, 0.72)
axa.invert_xaxis()
axa.set_xlabel('surfacing coverage (%)')
axa.set_ylabel('surfaced validity')
axa.legend(fontsize=6.5, frameon=False, loc='upper left')
axa.set_title('(a) reliability predictor', fontsize=8, loc='left')

# (b) 过滤器 risk-coverage (e106c 折内)
j = json.load(open(os.path.join(CACHE, 'e106b_certfilter.json'),
                   encoding='utf-8'))
cur = j['risk_coverage_foldnorm']
keep = [c['keep_frac'] * 100 for c in cur]
wrong = [c['wrong_rate'] * 100 for c in cur]
base = j['certified_wrong_rate_pooled'] * 100
axb.plot(keep, wrong, 'o-', color='#1a5276', lw=1.4, ms=3.4,
         label='keep most trusted')
axb.axhline(base, ls='--', color='#922b21', lw=1.0,
            label='no filter (%.1f%%)' % base)
axb.annotate('48.4% at 20%', xy=(20, 48.4), xytext=(38, 55),
             fontsize=6.5, color='#1a5276',
             arrowprops=dict(arrowstyle='->', color='#1a5276', lw=0.6))
axb.set_xlim(15, 103)
axb.set_ylim(40, 68)
axb.set_xlabel('retained fraction of perfectly stable windows (%)')
axb.set_ylabel('residual wrong rate (%)')
axb.legend(fontsize=6.5, frameon=False, loc='upper right')
axb.set_title('(b) perfectly-stable-window filter', fontsize=8, loc='left')

for ax in (axa, axb):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.4, h_pad=1.4)
fig.savefig(OUT)
print('saved', OUT, '| keep20 wrong %.1f base %.1f' % (wrong[-1], base))
