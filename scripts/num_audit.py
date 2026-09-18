# -*- coding: utf-8 -*-
"""关键数字审计: 正文声明 vs 表格文件 (机器可查部分)."""
import re

P = r'D:\0科研\工作1\第11篇SCI\paper'
tex = open(f'{P}\\INS_StabilityNotValidity.tex', encoding='utf-8').read()


def has(pat, label):
    ok = re.search(pat, tex) is not None
    print(f'  {"OK " if ok else "MISS"} {label}')


def table_has(fname, pat, label):
    t = open(f'{P}\\tables\\{fname}', encoding='utf-8').read()
    ok = re.search(pat, t) is not None
    print(f'  {"OK " if ok else "MISS"} {label} [{fname}]')


print('== 正文关键数字 ==')
has(r'hit@3 0\.25', 'TE max hit@3 0.25 (text)')
has(r'no method exceeds hit@3 0\.25', 'TE claim uses 0.25')
has(r'0\.629 vs\.\\ 0\.534|0\.629 vs\.\s*0\.534', 'GRAA 0.629 vs 0.534')
has(r'9\.5pp', 'GRAA +9.5pp')
has(r'0\.609.*0\.620|0\.620', 'GRAA 0.609 / PropRank 0.620')
has(r'\\lceil0\.236, 0\.269|0\.236, 0\.269', 'quadrant CI')
has(r'0\.501', 'mitigation single 0.501')

print('== 表格对应行 ==')
table_has('tab_main_te.tex', r'zDev & none & 0\.00 & 0\.25', 'TE zDev 0.25')
table_has('tab_main_scm.tex', r'GRAA & PCMCI-anom & 0\.37 & 0\.63',
          'SCM GRAA anom 0.63')
table_has('tab_main_scm.tex', r'GRAA & PCMCI-clean & 0\.37 & 0\.63',
          'SCM GRAA clean 0.63 (新行)')
table_has('tab_main_scm.tex', r'GRAA & true & 0\.34 & 0\.61',
          'SCM GRAA true 0.61')
table_has('tab_main_scm.tex', r'PropRank & true & 0\.37 & 0\.62',
          'SCM PropRank true 0.62')
table_has('tab_quadrant.tex', r'All methods & 0\.25', 'quadrant All 0.25')
table_has('tab_quadrant.tex', r'Unstable-wrong', '四象限列存在')
table_has('tab_swat.tex', r'PropRank & silver & 0\.47 & 0\.60',
          'SWaT PropRank silver 0.60')
table_has('tab_swat.tex', r'GCN-Rank & corr', 'SWaT GCN corr 行')
table_has('tab_swat.tex', r'GRAA\(p\{=\}0\.4\) & silver', 'SWaT GRAA 行')
table_has('tab_rcaeval.tex', r'AERec & -- & 0\.98 & 0\.99',
          'RCAEval AERec 0.99')
table_has('tab_predictor.tex', r'LOSO \(leaky\)', 'LOSO leaky 标注')
table_has('tab_graa_sens.tex', r'\\beta\{=\}0\.5', 'beta 表头')
table_has('tab_app_gnn.tex', r'0\.27 & 0\.31', 'GNN 附录 0.27/0.31')
table_has('tab_app_scorer.tex', r'Grad & 0\.51 & 0\.60',
          'scorer 附录 Grad SCM')

print('== 一致性对 ==')
t1 = open(f'{P}\\tables\\tab_main_scm.tex', encoding='utf-8').read()
print('  GRAA 3行齐全:', t1.count('& PCMCI-clean & 0\.37'.replace('\\.', '.')) >= 1
      and len(re.findall(r'^GRAA', t1, re.M)) == 3)
print('  敏感性 p=0 行 0.554:', '0.554' in
      open(f'{P}\\tables\\tab_graa_sens.tex', encoding='utf-8').read())
