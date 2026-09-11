# -*- coding: utf-8 -*-
"""SCI标准配色图表生成: GRAA柱状图 + 象限散点 + 退化曲线 + 覆盖曲线."""
import json, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Nature 封面提取配色 (2026-08-26 用户指定)
COLORS = {
    'blue': '#7FA4C8',    # Periwinkle Blue  — PropRank 系
    'cyan': '#DDE6F3',    # Pale Mist Blue   — 浅填充/次强调
    'green': '#8FB79B',   # Soft Mint Green  — DPTA-G 系
    'yellow': '#B59BC8',  # Lilac Lavender   — 次类别
    'red': '#D08AB6',     # Orchid Pink      — GRAA 高亮
    'purple': '#6A5A8F',  # Muted Violet     — GRAA 对照/深类别
    'grey': '#8E8E8E',    # 中性灰 (注释/参照, 非主题色)
    'darkblue': '#3F345B'  # Indigo Plum     — 文本/强调
}

FIG = '../paper/figures'
CACHE = '_cache'


def fig_graa_bar():
    """GRAA vs 基线柱状图(图4) — seed 级误差棒 + 显著性标注 (e25)."""
    e25 = json.load(open(f'{CACHE}/e25_graa_signif.json'))
    pm = e25['per_seed_means']

    def ms(key):
        return np.array(pm[key])

    entries = [
        ('PropRank\n(true)', 'true/PropRank', COLORS['blue']),
        ('PropRank\n(contam)', 'pcmci_anom/PropRank', COLORS['cyan']),
        ('GRAA\n(true)', 'true/GRAA(p=0.4)', COLORS['red']),
        ('GRAA\n(contam)', 'pcmci_anom/GRAA(p=0.4)', COLORS['purple']),
        ('DPTA-G\n(true)', 'true/DPTA-G', COLORS['green']),
        ('DPTA-G\n(contam)', 'pcmci_anom/DPTA-G', COLORS['yellow']),
        ('zDev', 'true/zDev', COLORS['grey']),
    ]
    labels = [e[0] for e in entries]
    means = [ms(e[1]).mean() for e in entries]
    errs = [ms(e[1]).std(ddof=1) for e in entries]
    colors = [e[2] for e in entries]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(len(labels)), means, yerr=errs, capsize=3,
                  color=colors, edgecolor='white', linewidth=0.5,
                  error_kw=dict(elinewidth=1, ecolor='#333333'))

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.022,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8,
                fontweight='bold')

    t = e25['tests']['pcmci_anom: GRAA vs PropRank']
    ax.annotate(f'+{t["delta"]*100:.1f}pp\np<0.001',
                xy=(1, means[1]), xytext=(3, means[1] - 0.045),
                fontsize=9, fontweight='bold', color=COLORS['red'],
                arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=1.5))

    ax.set_ylabel('hit@3 (5 seeds, ±1 SD)', fontsize=11)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.40, 0.72)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_graa_bar.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print('-> fig_graa_bar.pdf (error bars + significance)')


def fig_quadrant():
    """稳定性-正确性散点图(图1重制SCI配色)."""
    e1 = json.load(open(f'{CACHE}/e1_anchor.json'))
    cc = collections.defaultdict(list)
    for r in e1:
        if r.get('method', '_').startswith('_') or r.get('family') is None:
            continue
        if r['family'] != 'none' and r.get('acr3_base') is not None:
            cc[(r['dataset'], r['scorer'], r['method'], r['family'],
                r['strength'], r['graph_source'])].append((r['acr3_base'], r['hit3']))

    pts = [(np.mean([a for a, _ in v]), np.mean([h for _, h in v])) for v in cc.values()]

    fig, ax = plt.subplots(figsize=(5, 4.5))

    method_colors = {
        'DPTA-G': COLORS['blue'], 'PropRank': COLORS['red'],
        'GraphGranger': COLORS['green'], 'GlobalCF': COLORS['cyan'],
        'AERec': COLORS['purple'], 'Grad': COLORS['yellow'],
        'zDev': COLORS['grey']
    }

    # 按方法着色
    for (ds, sc, m, fam, st, src), vals in cc.items():
        a = np.mean([x[0] for x in vals])
        h = np.mean([x[1] for x in vals])
        color = method_colors.get(m, COLORS['darkblue'])
        ax.scatter(a, h, s=20, alpha=0.5, color=color, edgecolors='none')

    # 阈值线
    ax.axvline(0.8, color=COLORS['darkblue'], ls='--', lw=1, alpha=0.7)
    ax.axhline(0.4, color=COLORS['darkblue'], ls='--', lw=1, alpha=0.7)

    # 象限标签
    ax.text(0.92, 0.03, 'Stable-Wrong', fontsize=9, style='italic',
            color=COLORS['red'], fontweight='bold')
    ax.text(0.92, 0.92, 'Stable-Right', fontsize=9, style='italic', color=COLORS['green'])

    # 图例
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                          markersize=6, label=m) for m, c in method_colors.items()]
    ax.legend(handles=handles, fontsize=7, loc='lower left', framealpha=0.8)

    ax.set_xlabel('ACR@3 (Stability)', fontsize=11)
    ax.set_ylabel('hit@3 (Validity)', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_quadrant.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print('-> fig_quadrant.pdf (SCI colors)')


def fig_curves():
    """ACR退化曲线(图2重制SCI配色). 删除面板用 e27 九点合并网格."""
    a1 = json.load(open(f'{CACHE}/a1_analysis.json'))
    curves = a1['acr_curves']
    merged = json.load(open(f'{CACHE}/e27_merged.json'))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    fams = [('del', 'Edge Deletion'), ('add', 'Edge Addition'), ('rew', 'Rewiring')]

    for ax, (fam, title) in zip(axes, fams):
        for m, color in zip(('PropRank', 'GraphGranger', 'DPTA-G'),
                            (COLORS['red'], COLORS['green'], COLORS['blue'])):
            if fam == 'del':
                c = merged.get(f'{m}|true')
                xs, ys = c['xs'], c['acr']
                ax.plot(xs, c['hit'], 's--', color=color, markersize=3,
                        linewidth=1.0, alpha=0.75,
                        markeredgecolor='white', markeredgewidth=0.4)
            else:
                k = f'{m}|{fam}|true|iforest'
                if k not in curves:
                    continue
                c = curves[k]
                xs, ys = c['xs'], c['ys']
            ax.plot(xs, ys, 'o-', color=color, label=m, markersize=4,
                    linewidth=1.5, markeredgecolor='white', markeredgewidth=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r'Perturbation strength ($\epsilon$)', fontsize=9)
        ax.axhline(0.8, ls=':', color=COLORS['grey'], lw=0.8)
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel('ACR@3', fontsize=10)
    axes[0].legend(fontsize=8, framealpha=0.8)
    axes[0].set_title('Edge Deletion (solid ACR@3, dashed hit@3)',
                      fontsize=10)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_curves.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print('-> fig_curves.pdf (SCI colors)')


def fig_coverage():
    """覆盖-正确率曲线(图3重制SCI配色)."""
    e3 = json.load(open(f'{CACHE}/e3_results.json'))
    cov = e3['coverage']

    coverages = [c['coverage'] for c in cov]
    accuracies = [c['accuracy'] for c in cov]

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(coverages, accuracies, 'o-', color=COLORS['blue'], markersize=6,
            linewidth=1.5, markeredgecolor='white', markeredgewidth=0.5)

    base = accuracies[-1]
    ax.axhline(base, ls=':', color=COLORS['grey'], lw=1,
               label=f'No selection ({base:.2f})')

    # 标注关键点
    ax.annotate(f'({coverages[2]:.1f}, {accuracies[2]:.2f})',
                xy=(coverages[2], accuracies[2]),
                xytext=(coverages[2]+0.1, accuracies[2]+0.02),
                fontsize=8, arrowprops=dict(arrowstyle='->', color=COLORS['darkblue']))

    ax.set_xlabel('Coverage', fontsize=11)
    ax.set_ylabel('hit@3 on surfaced windows', fontsize=10)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_coverage.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print('-> fig_coverage.pdf (SCI colors)')


def fig_source_swap():
    """图源传导柱状图(新增图5)."""
    methods = ['PropRank', 'DPTA-G', 'GCN-Rank']
    sources = ['True', 'PCMCI\n(clean)', 'PCMCI\n(contam)', 'Corr-thr']
    # 从缓存提取
    e8 = json.load(open(f'{CACHE}/e8_grid.json'))
    agg = collections.defaultdict(list)
    for r in e8:
        agg[(r['method'], r['graph_source'])].append(r['hit3'])
    e10 = json.load(open(f'{CACHE}/e10_swat.json'))
    for r in e10:
        agg[(r['method'], r['graph_source'])].append(r['hit3'])

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(methods))
    width = 0.18
    src_colors = [COLORS['blue'], COLORS['cyan'], COLORS['red'], COLORS['yellow']]
    src_keys = ['true', 'pcmci', 'pcmci_anom', 'corr']
    src_labels = ['True/Silver', 'PCMCI clean', 'PCMCI contam', 'Correlation']

    for i, (sk, sl, c) in enumerate(zip(src_keys, src_labels, src_colors)):
        vals = []
        for m in methods:
            v = agg.get((m, sk), [])
            vals.append(np.mean(v) if v else 0)
        bars = ax.bar(x + i*width, vals, width, label=sl, color=c,
                      edgecolor='white', linewidth=0.5)

    ax.set_ylabel('hit@3', fontsize=11)
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(methods, fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_source_swap.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print('-> fig_source_swap.pdf')


if __name__ == '__main__':
    fig_graa_bar()
    fig_quadrant()
    fig_curves()
    fig_coverage()
    fig_source_swap()
    print('\nAll 5 figures generated with SCI colors')
