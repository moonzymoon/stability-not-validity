# -*- coding: utf-8 -*-
"""对比第10篇(JIIS)与第11篇(DKE)的写作风格量化指标。"""
import re
import statistics
import fitz


def extract(pdf):
    d = fitz.open(pdf)
    txt = ''
    for p in range(d.page_count):
        txt += d[p].get_text() + '\n'
    i = txt.find('Introduction')
    j = txt.rfind('References')
    if j < 0:
        j = len(txt)
    return txt[i:j]


def body_text(txt):
    txt = re.sub(r'[\x00-\x08\x0b-\x1f]', ' ', txt)
    # 去掉独立数字(行号/页码/图表号)
    txt = re.sub(r'^\s*\d{1,4}\s*$', '', txt, flags=re.M)
    return txt


def sentences(txt):
    # 简单分句: 句号/问号/叹号+空白; 缩写误切对两篇对称, 可比
    s = re.split(r'(?<=[.?!])\s+(?=[A-Z(])', txt)
    return [x for x in s if 4 <= len(x.split()) <= 120]


def stats(txt, name):
    words = len(re.findall(r'[A-Za-z][A-Za-z-]*', txt))
    sents = sentences(txt)
    lens = [len(s.split()) for s in sents]
    long35 = sum(1 for l in lens if l > 35) / len(lens) * 100
    long45 = sum(1 for l in lens if l > 45) / len(lens) * 100
    we = len(re.findall(r'\b[Ww]e\b', txt)) / words * 1000
    passive = len(re.findall(r'\b(is|are|was|were|be|been)\s+\w+(ed|en)\b', txt, re.I)) / words * 1000
    semis = txt.count(';') / words * 1000
    parens = txt.count('(') / words * 1000
    print(f'{name}: 词数{words} 句数{len(sents)} 均句长{statistics.mean(lens):.1f} 中位{statistics.median(lens):.0f} '
          f'>35词句{long35:.0f}% >45词句{long45:.0f}% we频率{we:.1f}/千词 被动式~{passive:.1f}/千词 分号{semis:.1f} 括号{parens:.1f}')
    return lens


p10 = body_text(open(r'D:/0科研/工作1/第11篇SCI/_p10_fulltext.txt', encoding='utf-8').read())
# 第10篇从 Introduction 到 References 前
p10 = p10[p10.find('Alarm management is a core'):]
p10 = p10[:p10.find('[1] Abdulaal')] if '[1] Abdulaal' in p10 else p10

p11 = body_text(extract(r'D:/0科研/工作1/第11篇SCI/paper/INS_StabilityNotValidity.pdf'))

l10 = stats(p10, '第10篇 JIIS')
l11 = stats(p11, '第11篇 DKE ')
print()
print('第10篇最长句词数:', max(l10), ' 第11篇最长句词数:', max(l11))
