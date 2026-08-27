# -*- coding: utf-8 -*-
"""从 _cache JSON 生成论文 LaTeX 表格与图 (红线: 正文数字必须可追溯到缓存)。
输出: paper/tables/*.tex, paper/figures/*.pdf
用法: python scripts/gen_tables.py [all|tab_main|tab_quadrant|fig_curves|...]
"""
import os
import sys
import json
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

CACHE = os.path.join(SRC, '_cache')
TAB = os.path.join(SRC, '..', 'paper', 'tables')
FIG = os.path.join(SRC, '..', 'paper', 'figures')
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

GRAPH_DEP = {'DPTA-G', 'PropRank', 'GraphGranger'}
METHOD_ORDER = ['DPTA-G', 'PropRank', 'GraphGranger', 'GlobalCF', 'AERec',
                'Grad', 'zDev', 'CondAttr', 'Random']


def load(fn):
    p = os.path.join(CACHE, fn)
    if not os.path.exists(p):
        print(f"[skip] {fn} not found")
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def fmt(x, nd=2):
    return '--' if x is None else f"{x:.{nd}f}"


def tab_main():
    """主表: SCM(中档) 与 TE, 方法x图源: hit@1/3/5 + ACR@3."""
    BSB = chr(92)  # backslash
    for tag, fn, srcs in (('scm', 'e2_scm_quality.json',
                           ('true', 'pcmci_clean', 'pcmci_anom')),
                          ('te', 'e2_te.json', ('pcmci',))):
        recs = load(fn)
        if recs is None:
            continue
        hit = collections.defaultdict(list)
        acr = collections.defaultdict(list)
        for r in recs:
            if (r.get('method', '').startswith('_')
                    or r.get('train_frac', 1.0) != 1.0):
                continue
            key = (r['method'], r['graph_source'], r['scorer'])
            if r['family'] == 'none':
                hit[key].append((r['hit1'], r['hit3'], r['hit5']))
            elif r.get('acr3_base') is not None:
                acr[key].append(r['acr3_base'])

        def boot_ci(vals, B=2000, seed=0):
            vals = np.asarray(vals, float)
            if len(vals) < 3:
                return None
            rng = np.random.default_rng(seed)
            bs = [vals[rng.integers(0, len(vals), len(vals))].mean()
                  for _ in range(B)]
            return np.percentile(bs, 2.5), np.percentile(bs, 97.5)

        L = []
        L.append(BSB + 'begin{table*}[t]' + BSB + 'centering' + BSB + 'footnotesize')
        if tag == 'scm':
            L.append(BSB + 'caption{Dual-metric grid on synthetic SCM (medium '
                    'difficulty, full scorer training): validity of the '
                    'base-graph attribution (hit@$K$) and stability under '
                    'perturbation of the same reference graph ($'
                    + BSB + 'mathrm{ACR}@3$, deployment column), mean over 5 '
                    'trajectories $' + BSB + 'times$ 3 scorers; brackets give '
                    'a bootstrap 95\\% CI over the 15 trajectory$'
                    + BSB + 'times$scorer cell means --- because each cell '
                    'already averages ${\\sim}70$ windows, this interval '
                    'reflects between-cell variance only and understates '
                    'total uncertainty; hypothesis-level claims in the text '
                    'use window- and seed-level paired tests. GRAA rows '
                    '(added post hoc) carry seed-level CIs over five '
                    'trajectories under the isolation-forest scorer. The '
                    'academic column reports '
                    'the ACR of perturbing the \\emph{true} graph (defined '
                    'once per method; it coincides with the deployment value '
                    'in the true-graph row and is constant across rows, '
                    'except GRAA whose pruned deployment graph changes the '
                    'perturbation base). '
                    'Graph-dependent methods are perturbed on the graph; '
                    'graph-free methods on the input.}')
        else:
            L.append(BSB + 'caption{Deployment scenario on the TE process '
                    '(41 measurements, PCMCI-learned graph, 6 fault units): '
                    'validity and stability under graph/input perturbation, '
                    'mean over 3 scorers; brackets give a bootstrap 95\\% '
                    'CI over the 6 units $' + BSB + 'times$ 3 scorer cell '
                    'means (between-cell variance only; see '
                    'Table~\\ref{tab:main-scm} caption).}')
        L.append(BSB + 'label{tab:main-' + tag + '}')
        L.append(BSB + 'setlength' + BSB + 'tabcolsep{3pt}')
        L.append(BSB + 'begin{tabular}{llccccc}')
        L.append(BSB + 'toprule')
        L.append('Method & Source & hit@1 & hit@3 (CI) & hit@5 & ACR@3 dep. & ACR@3 acad.' + BSB + BSB)
        L.append(BSB + 'midrule')
        # 学术参照: 每 (method) 用 true 场景的 ACR cell 值 (广播到所有行)
        acad = {}
        if tag == 'scm':
            for m in METHOD_ORDER:
                avals = []
                for sc in ('iforest', 'pca', 'ocsvm'):
                    if (m, 'true', sc) in acr:
                        avals.append(np.mean(acr[(m, 'true', sc)]))
                if avals:
                    acad[m] = float(np.mean(avals))
        for m in METHOD_ORDER:
            msrcs = srcs if m in GRAPH_DEP else ('none',)
            for src in msrcs:
                cells_by_sc = [sc for sc in ('iforest', 'pca', 'ocsvm')
                               if (m, src, sc) in hit]
                if not cells_by_sc:
                    continue
                h1c = [np.mean([h[0] for h in hit[(m, src, sc)]])
                       for sc in cells_by_sc]
                h3c = [np.mean([h[1] for h in hit[(m, src, sc)]])
                       for sc in cells_by_sc]
                h5c = [np.mean([h[2] for h in hit[(m, src, sc)]])
                       for sc in cells_by_sc]
                h1, h3, h5 = np.mean(h1c), np.mean(h3c), np.mean(h5c)
                ac = [np.mean(acr[(m, src, sc)]) for sc in cells_by_sc
                      if (m, src, sc) in acr]
                a_str = fmt(np.mean(ac)) if ac else '--'
                ci3 = boot_ci(h3c)

                def short(x):
                    return f"{x:.3f}"[1:] if 0 <= x < 1 else f"{x:.3f}"
                h3_str = fmt(h3) + (' [' + short(ci3[0]) + ',' + short(ci3[1])
                                    + ']' if ci3 else '')
                a_t_str = fmt(acad.get(m)) if acad.get(m) is not None else '--'
                sname = (src.replace('pcmci_clean', 'PCMCI-clean')
                         .replace('pcmci_anom', 'PCMCI-anom')
                         .replace('pcmci', 'PCMCI'))
                L.append(f"{m} & {sname} & {fmt(h1)} & {h3_str} & {fmt(h5)} & {a_str} & {a_t_str}" + BSB + BSB)
        # GRAA 行(防回归补丁): 从 e33 种子级缓存追加, 表注的 \dag 行
        if tag == 'scm':
            e33 = load('e33_graa_full.json')
            if e33:
                L.append(BSB + 'midrule')
                for src in ('true', 'pcmci_clean', 'pcmci_anom'):
                    h1 = e33[f'{src}/hit1']; h3 = e33[f'{src}/hit3']
                    h5 = e33[f'{src}/hit5']; ac = e33[f'{src}/acr']
                    sname = (src.replace('pcmci_clean', 'PCMCI-clean')
                             .replace('pcmci_anom', 'PCMCI-anom'))
                    L.append(f"GRAA($p{{=}}0.4$){BSB}dag & {sname} & "
                             f"{h1['mean']:.2f} & {h3['mean']:.2f} "
                             f"[{h3['lo']:.2f},{h3['hi']:.2f}] & "
                             f"{h5['mean']:.2f} & {ac['mean']:.2f} & "
                             f"{e33['true/acr']['mean']:.2f}" + BSB + BSB)
                L.append(BSB + 'midrule')
        L.append(BSB + 'bottomrule')
        L.append(BSB + 'end{tabular}')
        L.append(BSB + 'end{table*}')
        open(os.path.join(TAB, f'tab_main_{tag}.tex'), 'w',
             encoding='utf-8').write(chr(10).join(L))
        print(f'-> tab_main_{tag}.tex')


def tab_quadrant():
    a = load('a1_analysis.json')
    if a is None:
        return
    q = a['quadrant']
    lines = [r'\begin{table}[t]\centering\small',
             r'\caption{Quadrant decomposition of (ACR@3, hit@3) cells on '
             r'the SCM grid (2700 cells; thresholds ACR$>$0.8, '
             r'hit$<$0.4, $K{=}3$; each row sums to 1). Stable-wrong is '
             r'the decoupling evidence; the all-method share carries '
             r'bootstrap 95\% CI [0.236, 0.269]. The quadrant is '
             r'populated mainly by pseudo-stable families --- the '
             r'genuinely transmitting method escapes it under random '
             r'perturbation (PropRank 0.01) but not under source bias '
             r'(Sec.~source swap).}',
             r'\label{tab:quadrant}', r'\begin{tabular}{lcccc}',
             r'\toprule Method & Stable-wrong & Stable-right & '
             r'Unstable-right & Unstable-wrong\\\midrule']
    for m, qq in sorted(q['by_method'].items()):
        lines.append(f"{m} & {qq['stable_wrong']:.2f} & "
                     f"{qq['stable_right']:.2f} & "
                     f"{qq['unstable_right']:.2f} & "
                     f"{qq['unstable_wrong']:.2f}\\\\")
    lines += [r'\midrule All methods & ' +
              f"{q['all']['stable_wrong']:.2f} & "
              f"{q['all']['stable_right']:.2f} & "
              f"{q['all']['unstable_right']:.2f} & "
              f"{q['all']['unstable_wrong']:.2f}\\\\",
              r'\bottomrule', r'\end{tabular}', r'\end{table}']
    open(os.path.join(TAB, 'tab_quadrant.tex'), 'w', encoding='utf-8').write(
        '\n'.join(lines))
    print('-> tab_quadrant.tex')


def fig_curves():
    a = load('a1_analysis.json')
    if a is None:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    curves = a['acr_curves']
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, fam in zip(axes, ('del', 'add', 'rew')):
        for m, color in zip(('PropRank', 'GraphGranger', 'DPTA-G'),
                            ('C0', 'C1', 'C2')):
            k = f'{m}|{fam}|true|iforest'
            if k not in curves:
                continue
            c = curves[k]
            ax.plot(c['xs'], c['ys'], 'o-', color=color, label=m)
        ax.set_title(f'{fam} (true graph, iforest)')
        ax.set_xlabel('perturbation strength')
        ax.axhline(0.8, ls=':', c='gray', lw=0.8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('ACR@3')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_curves.pdf'))
    print('-> fig_curves.pdf')


def fig_quadrant():
    recs = load('e1_anchor.json')
    if recs is None:
        return
    cells = collections.defaultdict(list)
    for r in recs:
        if r.get('method', '').startswith('_') or r['family'] == 'none':
            continue
        if r.get('acr3_base') is None:
            continue
        cells[(r['dataset'], r['scorer'], r['method'], r['family'],
               r['strength'], r['graph_source'])].append((r['acr3_base'], r['hit3']))
    pts = [(np.mean([a for a, _ in v]), np.mean([h for _, h in v]), k[2])
           for k, v in cells.items()]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    markers = ['o', 's', '^', 'v', 'D', 'P', 'X', '*']
    for i, m in enumerate(('DPTA-G', 'PropRank', 'GraphGranger', 'GlobalCF',
                           'AERec', 'Grad', 'zDev', 'Random')):
        sub = [(a, h) for a, h, mm in pts if mm == m]
        if sub:
            ax.scatter([a for a, _ in sub], [h for _, h in sub], s=26,
                       alpha=0.55, color=f'C{i}', marker=markers[i], label=m,
                       edgecolors='k', linewidths=0.3)
    ax.axvline(0.8, ls=':', c='k', lw=0.8)
    ax.axhline(0.4, ls=':', c='k', lw=0.8)
    ax.set_xlabel('ACR@3 (stability)')
    ax.set_ylabel('hit@3 (validity)')
    ax.text(0.9, 0.05, 'stable-wrong', fontsize=9, style='italic')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_quadrant.pdf'))
    print('-> fig_quadrant.pdf')


def tab_predictor():
    a = load('e3_results.json')
    if a is None:
        return
    lines = [r'\begin{table}[t]\centering\small',
             r'\caption{Reliability predictor AUROC under LODO/LOSO with '
             r'feature-group ablations.}',
             r'\label{tab:predictor}', r'\begin{tabular}{llcc}',
             r'\toprule Model & Dropped groups & LODO & LOSO\\\midrule']
    for r in a['results']:
        if r['model'] != 'hgb':
            continue
        drop = ', '.join(r['drop']) if r['drop'] else '(none)'
        lines.append(f"HGB & {drop} & {r['summary'].get('LODO', float('nan')):.3f} & "
                     f"{r['summary'].get('LOSO', float('nan')):.3f}\\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    open(os.path.join(TAB, 'tab_predictor.tex'), 'w', encoding='utf-8').write(
        '\n'.join(lines))
    # coverage curve figure
    if a.get('coverage'):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        cov = [c['coverage'] for c in a['coverage']]
        acc = [c['accuracy'] for c in a['coverage']]
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        ax.plot(cov, acc, 'o-')
        base = acc[-1]
        ax.axhline(base, ls=':', c='gray', label=f'no selection ({base:.2f})')
        ax.set_xlabel('coverage')
        ax.set_ylabel('hit@3 on surfaced windows')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, 'fig_coverage.pdf'))
        print('-> tab_predictor.tex, fig_coverage.pdf')


def tab_mitigation():
    a = load('e4_mitigation.json')
    if a is None:
        return
    agg = collections.defaultdict(list)
    for r in a:
        agg[r['scenario']].append(r)
    lines = [r'\begin{table}[t]\centering\small',
             r'\caption{Mitigation: single graph vs.\ perturbation ensembles '
             r'(mean / rank / confidence-weighted) and oracle true graph.}',
             r'\label{tab:mitigation}', r'\begin{tabular}{llccccc}',
             r'\toprule Scenario & Method & Single & Ens-mean & Ens-rank & '
             r'Ens-conf & Oracle\\\midrule']
    for scen in ('random', 'systematic'):
        by_m = collections.defaultdict(list)
        for r in agg.get(scen, []):
            by_m[r['method']].append(r)
        for m, rs in sorted(by_m.items()):
            def mn(f):
                return np.mean([r[f] for r in rs])
            lines.append(f"{scen} & {m} & {mn('hit_single'):.2f} & "
                         f"{mn('hit_ens_mean'):.2f} & {mn('hit_ens_rank'):.2f} & "
                         f"{mn('hit_ens_conf'):.2f} & {mn('hit_oracle_true'):.2f}\\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    open(os.path.join(TAB, 'tab_mitigation.tex'), 'w', encoding='utf-8').write(
        '\n'.join(lines))
    print('-> tab_mitigation.tex')




def tab_relation():
    """M1: 关系型异常下的方法表现 (hit@3 + 象限)."""
    recs = load('e5_relation.json')
    if recs is None:
        return
    agg = collections.defaultdict(lambda: dict(h=[], a=[]))
    for r in recs:
        k = (r['method'], r['graph_source'] if r['graph_source'] != 'none' else '-')
        agg[k]['h'].append(r['hit3'])
        if r.get('acr3') is not None:
            agg[k]['a'].append(r['acr3'])
    BSB = chr(92)
    L = [BSB + 'begin{table}[t]' + BSB + 'centering' + BSB + 'small',
         BSB + 'caption{Relational anomalies (VAR edge-coefficient drift; '
         'noise terms preserved in all nodes, marginals changed only '
         'secondarily): attribution validity by method, with stability '
         'where defined. The method ranking reorganizes relative to '
         'deviational anomalies (reconstruction $+31$ points, propagation '
         '$-29$); the stable-wrong quadrant is occupied by the '
         'regression-ablation family (GraphGranger: ACR 0.85--0.87 with '
         'hit@3 $' + BSB + 'approx$0.38).}',
         BSB + 'label{tab:relation}',
         BSB + 'begin{tabular}{llcc}', BSB + 'toprule',
         'Method & Source & hit@3 & ACR@3' + BSB + BSB, BSB + 'midrule']
    order = ['AERec', 'Grad', 'DPTA-G', 'zDev', 'GraphGranger',
             'GlobalCF', 'CondAttr', 'PropRank', 'Random']
    for m in order:
        for k in sorted([x for x in agg if x[0] == m]):
            v = agg[k]
            h = float(np.mean(v['h']))
            a = float(np.mean(v['a'])) if v['a'] else None
            L.append(f"{m} & {k[1]} & {h:.2f} & " +
                     (f"{a:.2f}" if a is not None else '--') + BSB + BSB)
    L += [BSB + 'bottomrule', BSB + 'end{tabular}', BSB + 'end{table}']
    open(os.path.join(TAB, 'tab_relation.tex'), 'w', encoding='utf-8').write(
        chr(10).join(L))
    print('-> tab_relation.tex')


def tab_alpha():
    """M5: DPTA-G alpha 扫描."""
    recs = load('e5_alpha.json')
    if recs is None:
        return
    agg = collections.defaultdict(lambda: dict(h=[], a=[]))
    for r in recs:
        agg[(r['alpha'], r['graph_source'])]['h'].append(r['hit3'])
        agg[(r['alpha'], r['graph_source'])]['a'].append(r['acr3'])
    BSB = chr(92)
    L = [BSB + 'begin{table}[t]' + BSB + 'centering' + BSB + 'small',
         BSB + 'caption{DPTA-G fusion-weight sweep ($' + BSB + 'alpha$ = '
         'deviation-path weight; $1-' + BSB + 'alpha$ = relation path): '
         'pseudo-stability is structural, not a parameter artifact --- even '
         'the pure relation path ($' + BSB + 'alpha=0$) is highly stable '
         'under graph perturbation.}',
         BSB + 'label{tab:alpha}',
         BSB + 'begin{tabular}{lcccc}', BSB + 'toprule',
         BSB + 'multirow{2}{*}{$' + BSB + 'alpha$} & '
         + BSB + 'multicolumn{2}{c}{true graph} & '
         + BSB + 'multicolumn{2}{c}{PCMCI-anom graph}' + BSB + BSB,
         BSB + 'cmidrule(lr){2-3}' + BSB + 'cmidrule(lr){4-5}',
         ' & hit@3 & ACR@3 & hit@3 & ACR@3' + BSB + BSB, BSB + 'midrule']
    for al in (0.0, 0.3, 0.5, 0.7, 1.0):
        row = [f"{al:.1f}"]
        for src in ('true', 'anom'):
            v = agg.get((al, src))
            row.append(f"{np.mean(v['h']):.2f}" if v else '--')
            row.append(f"{np.mean(v['a']):.2f}" if v and v['a'] else '--')
        L.append(' & '.join(row) + BSB + BSB)
    L += [BSB + 'bottomrule', BSB + 'end{tabular}', BSB + 'end{table}']
    open(os.path.join(TAB, 'tab_alpha.tex'), 'w', encoding='utf-8').write(
        chr(10).join(L))
    print('-> tab_alpha.tex')


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', 'tab_main'):
        tab_main()
    if which in ('all', 'tab_quadrant'):
        tab_quadrant()
    if which in ('all', 'fig_curves'):
        fig_curves()
    if which in ('all', 'fig_quadrant'):
        fig_quadrant()
    if which in ('all', 'tab_predictor'):
        tab_predictor()
    if which in ('all', 'tab_mitigation'):
        tab_mitigation()
    if which in ('all', 'tab_relation'):
        tab_relation()
    if which in ('all', 'tab_alpha'):
        tab_alpha()
