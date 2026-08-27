# -*- coding: utf-8 -*-
"""阶段1: DeepSeek/豆包反馈 + 指南 合规深度复核 (第1遍)."""
import re

t = open(r'D:\0科研\工作1\第11篇SCI\paper\DKE_StabilityNotValidity.tex',
         encoding='utf-8').read()
tab_rt = open(r'D:\0科研\工作1\第11篇SCI\paper\tables\tab_runtime.tex',
              encoding='utf-8').read()
tab_te = open(r'D:\0科研\工作1\第11篇SCI\paper\tables\tab_te_robust.tex',
              encoding='utf-8').read()
main = t.split('\\appendix')[0]

items = {
 'DS-P0-1 实例差正文': ('41/350' in t and '47/350' in t),
 'DS-P0-1 实例差图注': ('independently seeded perturbation instances' in t),
 'DS-P0-2 硬件': ('i7-12700F' in tab_rt),
 'DS-P0-2 GRAA伪代码': ('\\begin{algorithm}' in t),
 'DS-P0-3 超参声明': ('SCM medium configuration alone' in t),
 'DS-P0-4 作者信息': ('Baoding University' in t and 'liyanling@bdu.edu.cn' in t),
 'DS-P0-5 敏感表下沉': ('\\input{tables/tab_graa_sens}' not in main),
 'DS-P1-6 外部基线AERCA': ('AERCA~\\cite{han2025aerca}' in t and 'mid-pack' in t),
 'DS-P1-7 失败案例': ('Anatomy of one stable-wrong window' in t),
 'DS-P1-8 文献量33': (len(re.findall(r'bibitem', open(
     r'D:\0科研\工作1\第11篇SCI\paper\DKE_StabilityNotValidity.bbl',
     encoding='utf-8').read())) == 33),
 '豆包-表注重叠说明': ('overlapping sliding windows' in tab_te),
 '豆包-Vitae草稿': ('Vitae' in open(
     r'D:\0科研\工作1\第11篇SCI\04_投稿准备\TitlePage_Declarations.txt',
     encoding='utf-8').read()),
}
guide = {
 '行号': ('\\linenumbers' in t),
 '单盲作者': ('Yanling Li' in t),
 'CRediT': ('CRediT' in t),
 'GenAI声明': ('generative AI and AI-assisted' in t),
 '利益冲突': ('competing interest' in t),
 '资助': ('Funding' in t),
 '数据声明': ('Data availability' in t),
 '附录A.1': ('\\setcounter{table}{0}' in t),
 '数字引用': ('elsarticle-num' in t),
}
absw = len(re.sub(r'\\[a-zA-Z]+|[{}~$]', ' ',
    re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', t, re.S)
    .group(1)).split())
guide['摘要词数'] = (200 <= absw <= 250)
guide['关键词6个'] = (t.count(' \\sep ') == 5)

bad = [k for k, v in {**items, **guide}.items() if not v]
print(f'abstract words = {absw}')
for k, v in {**items, **guide}.items():
    print(('PASS ' if v else 'MISS ') + k)
print('\n未过:', bad if bad else '全部通过')
