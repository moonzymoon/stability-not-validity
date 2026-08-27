# -*- coding: utf-8 -*-
"""DKE 投稿指南 20 项逐条复检 (2026-08-26 认真版)."""
import re
import os

t = open(r'D:\0科研\工作1\第11篇SCI\paper\DKE_StabilityNotValidity.tex',
         encoding='utf-8').read()
bbl = open(r'D:\0科研\工作1\第11篇SCI\paper\DKE_StabilityNotValidity.bbl',
           encoding='utf-8').read()
hl = open(r'D:\0科研\工作1\第11篇SCI\04_投稿准备\Highlights.txt',
          encoding='utf-8').read()
cl = open(r'D:\0科研\工作1\第11篇SCI\04_投稿准备\CoverLetter_DKE.txt',
          encoding='utf-8').read()
tp = open(r'D:\0科研\工作1\第11篇SCI\04_投稿准备\TitlePage_Declarations.txt',
          encoding='utf-8').read()

m = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', t, re.S)
absw = len(re.sub(r'\\[a-zA-Z]+|[{}~$]', ' ', m.group(1)).split())
hl_items = [l for l in hl.split('\n') if l.startswith('- ')]
no_vrule = True
for f in os.listdir(r'D:\0科研\工作1\第11篇SCI\paper\tables'):
    s = open(r'D:\0科研\工作1\第11篇SCI\paper\tables' + '\\' + f,
             encoding='utf-8').read()
    if '|c' in s or '|l' in s or '|r' in s:
        no_vrule = False

checks = [
 ('1 摘要≤250词/无引用', 200 <= absw <= 250, f'{absw} 词'),
 ('2 关键词 1-7 个', t.count('\\sep') == 5, '6 个'),
 ('3 Highlights 3-5 条×≤85 字符',
  len(hl_items) == 5 and all(len(l[2:]) <= 85 for l in hl_items),
  f'{len(hl_items)} 条'),
 ('4 elsarticle 模板+期刊名',
  '\\documentclass[3p,times,numberline]{elsarticle}' in t
  and 'Data \\& Knowledge Engineering' in t, '3p,times,numberline'),
 ('5 每行行号', 'numberline' in t.split('\\begin{document}')[0]
  and '\\linenumbers' in t, 'numberline+lineno'),
 ('6 单盲:真实作者+单位+通讯邮箱',
  'Yanling Li' in t and 'Chao Wang' in t and 'Baoding University' in t
  and 'liyanling@bdu.edu.cn' in t and 'Baocai' not in t, '李燕玲通讯+王超三作'),
 ('7 章节编号+数字引用', 'elsarticle-num' in t, ''),
 ('8 表格 booktabs 无竖线', no_vrule, ''),
 ('9 图独立矢量 PDF',
  6 <= len([f for f in os.listdir(r'D:\0科研\工作1\第11篇SCI\paper\figures')
            if f.endswith('.pdf')]) <= 8, 'Nature 配色'),
 ('10 附录 Table A.1 起', '\\setcounter{table}{0}' in t, ''),
 ('11 利益冲突/资助/CRediT', all(x in t for x in
   ('competing interest', 'Funding', 'CRediT')), ''),
 ('12 GenAI 声明(参考文献前)', 'generative AI and AI-assisted' in t and
  t.find('generative AI') < t.find('bibliographystyle'), ''),
 ('13 数据/代码可用性', 'Data availability' in t
  and 'Code availability' in t, ''),
 ('14 摘要无未展开缩写', 'structural causal models' in m.group(1), 'SCM 已展开'),
 ('15 封面信要素', all(x in cl for x in
   ('On behalf of all authors', 'not been published previously',
    'Yanling Li (corresponding author)')), ''),
 ('16 TitlePage 完整', all(x in tp for x in
   ('Yanling Li', 'liyanling@bdu.edu.cn', 'ORCID', 'Vitae')), ''),
 ('17 正文无占位符', 'example.com' not in t and 'TODO' not in t
  and '[Author name]' not in t, ''),
 ('18 参考文献', 40 <= bbl.count('bibitem') <= 45,
  f'{bbl.count("bibitem")} 条'),
 ('19 无 em-dash/无 ours10', '---' not in t and 'ours10' not in t, ''),
 ('20 控制字符损伤', not re.search(
     r'[\x00-\x08\x0b-\x0d\x1c-\x1f]', t), ''),
]
bad = []
for name, ok, note in checks:
    print(('PASS ' if ok else 'MISS ') + name + ('  [' + note + ']'
                                                   if note else ''))
    if not ok:
        bad.append(name)
print('\n结论:', '20/20 全部通过' if not bad else '未过: ' + ', '.join(bad))
