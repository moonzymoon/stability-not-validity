# -*- coding: utf-8 -*-
"""P4 section 文件拼接后与 P11 重扫。"""
import re
import glob
import fitz
import sys

sys.path.insert(0, r'D:/0科研/工作1/第11篇SCI/src/scripts')
from orthogonality_scan import norm_words, shingles

t = ''
for f in sorted(glob.glob(r'D:/0科研/工作1/第4篇SCI/paper/sec*.tex')):
    t += open(f, encoding='utf-8', errors='ignore').read() + '\n'
t = re.sub(r'(?<!\\)%.*', '', t)
t = re.sub(r'\\begin\{(equation|table|figure|algorithm)\*?\}.*?\\end\{\1\*?\}', ' ', t, flags=re.S)
t = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?', ' ', t)
t = re.sub(r'[{}&$~^_\\]', ' ', t)
w = norm_words(t)
s = shingles(w)

d11 = fitz.open(r'D:/0科研/工作1/第11篇SCI/paper/DKE_StabilityNotValidity.pdf')
t11 = ''
for p in range(d11.page_count):
    t11 += d11[p].get_text() + '\n'
i = t11.find('Introduction')
t11 = t11[i:t11.rfind('References')]
s11 = shingles(norm_words(t11))

ov = s11 & s
print(f'P4正文词数{len(w)}, 与P11重叠shingle {len(ov)} ({len(ov)/len(s11)*100:.2f}%)')
for sh in list(ov)[:4]:
    print('  样例:', ' '.join(sh))
