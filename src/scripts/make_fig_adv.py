# -*- coding: utf-8 -*-
"""对抗 vs 随机扰动的小图(Nature 配色, 与 make_sci_figures 同风格)."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {'blue': '#7FA4C8', 'red': '#D08AB6', 'purple': '#6A5A8F',
          'grey': '#8E8E8E', 'darkblue': '#3F345B'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.linewidth': 0.8})

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
CACHE = os.path.join(SRC, '_cache')
FIG = os.path.join(SRC, '..', 'paper', 'figures')

d = json.load(open(os.path.join(CACHE, 'e59_adversarial.json'),
                   encoding='utf-8'))
conds = [('acr_rand_q01', 'Random\n$q{=}0.1$'),
         ('acr_rand_q02', 'Random\n$q{=}0.2$'),
         ('acr_adv_B2', 'Adversarial\n$B{=}2$'),
         ('acr_adv_B4', 'Adversarial\n$B{=}4$')]
fig, ax = plt.subplots(figsize=(5.2, 3.4))
x = np.arange(len(conds))
w = 0.36
for off, m, c in ((-w / 2, 'PropRank', COLORS['red']),
                  (w / 2, 'DPTA-G', COLORS['purple'])):
    sub = [r for r in d if r['method'] == m]
    means = [np.mean([r[k] for r in sub]) for k, _ in conds]
    sds = [np.std([r[k] for r in sub]) for k, _ in conds]
    ax.bar(x + off, means, w, yerr=sds, capsize=3, color=c,
           label=m, edgecolor='white', linewidth=0.5, error_kw=dict(lw=1))
    for xi, v in zip(x + off, means):
        ax.text(xi, v + 0.045, f'{v:.2f}', ha='center', va='bottom',
                fontsize=7.5, fontweight='bold', color='#333333')
ax.set_ylabel('ACR@3 (5 seeds, ±1 SD)', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels([lab for _, lab in conds], fontsize=8.5)
ax.set_ylim(0, 1.12)
ax.legend(frameon=False, fontsize=9, loc='lower left')
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
out = os.path.join(FIG, 'fig_adv.pdf')
fig.savefig(out, bbox_inches='tight')
print('saved', out)
