# -*- coding: utf-8 -*-
"""补充材料 Figure S1: 单窗口案例 (来自 e81_case_study.json 缓存)."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(HERE), '..', 'paper', 'figures')

d = json.load(open(os.path.join(HERE, '..', '_cache',
                                'e81_case_study.json')))
n = len(d['aerec_scores'])
idx = np.arange(n)

fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True)
for ax, key, name, sub in (
        (axes[0], 'aerec_scores', 'AERec',
         'stable-wrong: top-3 misses the root,\n'
         'window consistency %.2f' % d['aerec_window_consistency']),
        (axes[1], 'zdev_scores', 'zDev', 'correct: top-3 contains the root')):
    scores = np.array(d[key])
    roots = set(d['true_roots'])
    t3 = set(d[name.lower() + '_top3'])
    colors = ['#D08AB6' if i in roots else
              '#7FA4C8' if i in t3 else '#C9C9C9' for i in idx]
    ax.barh(idx, scores, color=colors, height=0.72)
    ax.invert_yaxis()
    ax.set_yticks(idx)
    ax.set_yticklabels([('$v_{%d}$' % i) + ('*' if i in roots else '')
                        for i in idx], fontsize=7.5)
    ax.set_title('%s --- %s' % (name, sub), fontsize=8.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=7.5)
    ax.set_xlabel('attribution score', fontsize=8.5)
fig.text(0.012, 0.015,
         'SCM seed 0 (medium), window %d, isolation-forest scorer; '
         'true root $v_6$ (pink); top-3 outlined blue vs grey.'
         % d['window_index'], fontsize=7.5, color='#555555')
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(os.path.join(FIG, 'fig_case.pdf'), dpi=300,
            bbox_inches='tight')
print('-> figures/fig_case.pdf')
