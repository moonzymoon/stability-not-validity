# -*- coding: utf-8 -*-
"""为已有官方 DOI 的文献条目批量补 doi 字段(仅高置信度规范 DOI)。"""
import re

P = r'D:/0科研/工作1/第11篇SCI/paper/references.bib'
t = open(P, encoding='utf-8').read()

DOIS = {
    'downs1993': '10.1016/0098-1354(93)80018-I',
    'liu2008iforest': '10.1109/ICDM.2008.17',
    'granger1969': '10.2307/1912791',
    'goh2017swat': '10.1007/978-3-319-71368-7_8',
    'guidotti2018survey': '10.1145/3236009',
    'yepmo2022anomaly': '10.1016/j.datak.2021.101946',
    'krishna2022disagreement': '10.1145/3531146.3533082',
    'chandola2009survey': '10.1145/1541880.1541882',
    'audibert2020usad': '10.1145/3394486.3403392',
    'davis2006prc': '10.1145/1143844.1143874',
    'pearl2009causality': '10.1017/CBO9780511803161',
    'breunim2000lof': '10.1145/335191.335388',
    'goldstein2016survey': '10.1007/978-3-319-31753-3_27',
    'hundman2018spacecraft': '10.1145/3219819.3219845',
    'runge2019inferring': '10.5194/essd-11-1941-2019',
    'keogh2003need': '10.1023/B:DAMI.0000009205.71541.bd',
    'dau2019ucr': '10.1109/JAS.2019.1911747',
}

n = 0
for key, doi in DOIS.items():
    pat = re.compile(r'(@\w+\{' + key + r',)(.*?)(\n\})', re.S)
    m = pat.search(t)
    if not m or 'doi' in m.group(2).lower():
        continue
    body = m.group(2)
    # 在 year 行后插入 doi
    new_body = re.sub(r'(\n  year\s*=\s*\{\d+\},?)', r'\1\n  doi = {' + doi + '}',
                      body, count=1)
    if new_body == body:
        new_body = body + '\n  doi = {' + doi + '}'
    t = t[:m.start(2)] + new_body + t[m.end(2):]
    n += 1

open(P, 'w', encoding='utf-8').write(t)
print(f'已补 {n} 条 DOI')

# 复核
entries = re.findall(r'@(\w+)\{([^,]+),(.*?)\n\}', t, re.S)
has = sum(1 for _, _, b in entries if 'doi' in b.lower())
print(f'现总数 {len(entries)} | 有DOI {has} | 无DOI {len(entries)-has}')
