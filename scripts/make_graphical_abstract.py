# -*- coding: utf-8 -*-
"""Graphical Abstract (submission asset, not a paper figure). v2:
fixes vision-review issues (label clearance, clipping, threshold
labels, panel balance)."""
import collections
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '..', '_cache')
OUT = os.path.join(HERE, '..', '..', '05_投稿包', 'Graphical_Abstract')

BLUE, RED, GREY, DARK = '#7FA4C8', '#D08AB6', '#8E8E8E', '#3F345B'

e1 = json.load(open(os.path.join(CACHE, 'e1_anchor.json')))
cc = collections.defaultdict(list)
for r in e1:
    if r.get('method', '_').startswith('_') or r.get('family') is None:
        continue
    if r['family'] != 'none' and r.get('acr3_base') is not None:
        cc[(r['dataset'], r['scorer'], r['method'], r['family'],
            r['strength'], r['graph_source'])].append(
                (r['acr3_base'], r['hit3']))
pts = [(np.mean([a for a, _ in v]), np.mean([h for _, h in v]))
       for v in cc.values()]

fig, axes = plt.subplots(1, 2, figsize=(13.3, 5.0),
                         gridspec_kw={'width_ratios': [1.8, 1]})

# ---- left: stability-validity plane ----
ax = axes[0]
ax.axvspan(0.8, 1.03, ymin=0, ymax=0.4 / 1.05, color=RED, alpha=0.18,
           zorder=0)
ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=15, color=BLUE,
           alpha=0.7, edgecolors='none', zorder=2)
ax.axvline(0.8, color=GREY, lw=1, ls='--', zorder=1)
ax.axhline(0.4, color=GREY, lw=1, ls='--', zorder=1)
ax.text(0.802, 1.006, 'ACR@3 = 0.8', fontsize=9, color=GREY, va='bottom')
ax.text(0.455, 0.407, 'hit@3 = 0.4', fontsize=9, color=GREY, va='bottom')
ax.set_xlabel('Stability  (perturbation consistency, ACR@3)', fontsize=12)
ax.set_ylabel('Validity  (correctness, hit@3)', fontsize=12)
ax.set_xlim(0.45, 1.03)
ax.set_ylim(-0.03, 1.08)
ax.set_title('Stable  ≠  correct', fontsize=15, color=DARK, pad=12)
# 指向阴影区内一个具体点簇
tgt = [p for p in pts if p[0] > 0.85 and p[1] < 0.25]
tx, ty = (np.mean([p[0] for p in tgt]), np.mean([p[1] for p in tgt])) \
    if tgt else (0.9, 0.15)
ax.annotate('stable-wrong:\n19–25% of the grid\n(36% under weak anomalies)',
            xy=(tx, ty), xytext=(0.56, 0.16), fontsize=11.5, color=DARK,
            arrowprops=dict(arrowstyle='-|>', color=DARK, lw=1.6,
                            shrinkA=2, shrinkB=4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=10)

# ---- right: ARP selective disclosure frontier ----
ax = axes[1]
cov = [100, 50, 30]
val = [0.34, 0.57, 0.67]
ax.plot(cov, val, marker='o', ms=9, lw=2.2, color=DARK, zorder=3)
offs = {100: (-10, 14), 50: (0, 14), 30: (-6, -34)}
labs = {100: 'no check: 0.34', 50: '50% surfaced: 0.57',
        30: '30% surfaced: 0.67'}
for x, y in zip(cov, val):
    ax.annotate(labs[x], (x, y), textcoords='offset points',
                xytext=offs[x], fontsize=10.5, color=DARK, ha='center',
                zorder=4)
ax.set_xlim(6, 124)
ax.set_ylim(0.20, 0.85)
ax.set_yticks([0.2, 0.4, 0.6, 0.8])
ax.set_xlabel('Windows surfaced (%)   →  stricter pre-check', fontsize=12)
ax.set_ylabel('Validity of surfaced set', fontsize=12)
ax.set_title('ARP pre-check: fewer, truer attributions', fontsize=12.5,
             color=DARK, pad=12)
ax.invert_xaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=10)

fig.suptitle('Stability does not certify validity in root-cause attribution',
             fontsize=16, color=DARK, y=1.03)
fig.tight_layout()
fig.savefig(OUT + '.png', dpi=150, bbox_inches='tight', facecolor='white')
fig.savefig(OUT + '.pdf', bbox_inches='tight', facecolor='white')
from PIL import Image
im = Image.open(OUT + '.png')
print('->', OUT + '.png', im.size)
