# -*- coding: utf-8 -*-
"""统计论文词数：正文（不含表格/公式/参考文献）+ 表格 + 标题抽取。"""
import re
import glob

t = open('paper/DKE_StabilityNotValidity.tex', encoding='utf-8').read()

t = re.sub(r'(?<!\\)%.*', '', t)

body = t[t.find(r'\begin{document}'):]
s = body
s = re.sub(r'\\begin\{(equation|table|figure|algorithm|verbatim|lstlisting)\*?\}.*?\\end\{\1\*?\}', ' ', s, flags=re.S)
s = re.sub(r'\\(?:input|include)\{[^}]*\}', ' ', s)
s = re.sub(r'\\bibliography\{[^}]*\}.*', ' ', s, flags=re.S)  # 参考文献之后不算
s = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?', ' ', s)
s = re.sub(r'[{}&$~^_\\]', ' ', s)
words = [w for w in s.split() if re.search(r'[A-Za-z]', w)]
print('正文词数(不含表格/公式/参考文献):', len(words))

tbl = ''
for f in glob.glob('paper/tables/*.tex'):
    tbl += open(f, encoding='utf-8').read()
tbl = re.sub(r'(?<!\\)%.*', '', tbl)
tbl = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?', ' ', tbl)
tbl = re.sub(r'[{}&$~^_\\]', ' ', tbl)
tw = [w for w in tbl.split() if re.search(r'[A-Za-z]', w)]
print('表格词数:', len(tw), '-> 正文+表格:', len(words) + len(tw))

m = re.search(r'\\title\{(.*?)\}\s*\n', t, flags=re.S)
if m:
    print('TITLE:', ' '.join(m.group(1).split()))
# 首段式标题（若 \title 带换行）
mm = re.search(r'\\title\{([^{}]*?:[^{}]*)', t)
if mm:
    print('TITLE_MAIN:', ' '.join(mm.group(1).split()))
