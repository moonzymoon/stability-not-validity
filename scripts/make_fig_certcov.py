# -*- coding: utf-8 -*-
"""证书覆盖率图 (e39): (a) 覆盖率随扰动强度; (b) 持证错率 + 覆盖样本数."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {'blue': '#7FA4C8', 'red': '#D08AB6', 'grey': '#8E8E8E',
          'darkblue': '#3F345B', 'purple': '#6A5A8F'}
FIG = '../paper/figures'

d = json.load(open('_cache/e39_certcov.json'))
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))

ax = axes[0]
for m, c, mk in (('DPTA-G', COLORS['blue'], 'o'),
                 ('PropRank', COLORS['red'], 's')):
    qs = [r['q'] for r in d['curves'][m]]
    cov = [100 * r['coverage'] for r in d['curves'][m]]
    ax.plot(qs, cov, marker=mk, color=c, label=m, lw=1.8, ms=5)
ax.set_xlabel('Perturbation strength $q$')
ax.set_ylabel('Certificate coverage (\\%)')
ax.set_ylim(-2, 30)
ax.legend(frameon=False, fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('(a) $\\Pr[\\gamma > 2\\eta]$ by strength', fontsize=10)

ax = axes[1]
cur = d['curves']['DPTA-G']
qs = [r['q'] for r in cur]
wr = [100 * r['certified_wrong'] for r in cur]
nc = [r['n_certified'] for r in cur]
ax.plot(qs, wr, marker='o', color=COLORS['blue'], lw=1.8, ms=5,
        label='certified-wrong rate')
for x, y, n in zip(qs, wr, nc):
    ax.annotate(f'$n$={n}', (x, y), textcoords='offset points',
                xytext=(6, 5), fontsize=8, color=COLORS['darkblue'])
# headline 参考线从缓存读值 (q=0.2 的 certified_wrong), 不手打
_head = [r for r in cur if r['q'] == 0.2][0]
_hw = round(100 * _head['certified_wrong'])
_hn = _head['n_certified']
ax.axhline(_hw, ls='--', lw=1, color=COLORS['grey'])
ax.annotate(f'{_hw}\\% (headline, n={_hn})', (0.055, _hw + 1.5),
            fontsize=8, color=COLORS['darkblue'])
ax.set_xlabel('Perturbation strength $q$')
ax.set_ylabel('Certified-wrong rate (\\%)')
ax.set_ylim(20, 60)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('(b) DPTA-G certified windows', fontsize=10)

fig.tight_layout()
fig.savefig(f'{FIG}/fig_certcov.pdf', dpi=300, bbox_inches='tight')
print('-> figures/fig_certcov.pdf')
