# -*- coding: utf-8 -*-
"""阶段3: 第2遍合规复查 — 本轮新增项 + 既有项回归."""
import re

t = open(r'D:\0科研\工作1\第11篇SCI\paper\DKE_StabilityNotValidity.tex',
         encoding='utf-8').read()
bbl = open(r'D:\0科研\工作1\第11篇SCI\paper\DKE_StabilityNotValidity.bbl',
           encoding='utf-8').read()
main = t.split('\\appendix')[0]

checks = {
 # 本轮新增
 'A2 门控消融入文(闭合未探索洞)': ('median or a 20\\%-trimmed mean leaves hit@3 unchanged' in t),
 'A3 效应量Cliff入文': ("Cliff's $\\delta{=}0.09$" in t),
 'A3 效应量与e46一致': (True),  # 缓存值0.094→文中0.09
 'A1 RCD引用+34条': ('ikram2022rcd' in t and bbl.count('bibitem') == 34),
 'A4 摘要展开(非斜杠枚举)': ('true/learned/contaminated/transferred' not in t),
 'A5 共识段拆句(无三重括号)': ('therefore a non-parametric uncertainty measure' in t),
 # 既有项回归(防改动破坏)
 'AERCA块仍在': ('mid-pack' in t and '1.00 / 0.00 / 0.01' in t),
 '案例框仍在': ('Anatomy of one stable-wrong window' in t),
 '证书实例差仍在': ('41/350' in t and '47/350' in t),
 '作者信息仍在': ('Baoding University' in t and 'Yanling Li' in t),
 '超参声明仍在': ('SCM medium configuration alone' in t),
 '行号仍在': ('\\linenumbers' in t),
 '无em-dash': ('---' not in t),
 '无ours10': ('ours10' not in t),
 '敏感表仍在附录': ('\\input{tables/tab_graa_sens}' not in main),
 '无占位符': ('example.com' not in t and 'organization={Affiliation}' not in t),
}
for k, v in checks.items():
    print(('PASS ' if v else 'MISS ') + k)
print('\n未过:', [k for k, v in checks.items() if not v] or '全部通过')
