# -*- coding: utf-8 -*-
"""提取块4: 实验文字 (737-1583), 去图环境与表input."""
import re

t = open(r'D:\0科研\工作1\第11篇SCI\paper\INS_StabilityNotValidity.tex',
         encoding='utf-8').read()
lines = t.split('\n')
seg = '\n'.join(lines[736:1583])
seg = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}\n?', '', seg, flags=re.S)
seg = re.sub(r'\\input\{tables/[^}]*\}\n?', '', seg)
open(r'D:\0科研\工作1\第11篇SCI\block4_tmp.txt', 'w',
     encoding='utf-8').write(seg)
print(len(seg.split('\n')), 'lines,', len(seg), 'chars')
