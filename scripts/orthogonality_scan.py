# -*- coding: utf-8 -*-
"""P11 与 P4/P5/P10 的文本重叠扫描(自我查重): 8词 shingle 重叠率。"""
import re
import fitz


def norm_words(text):
    text = re.sub(r'[\x00-\x08\x0b-\x1f]', ' ', text)
    text = re.sub(r'^\s*\d{1,4}\s*$', '', text, flags=re.M)  # 行号/页码
    text = re.sub(r'\[[0-9,]+\]', '', text)                  # 引用编号
    text = re.sub(r'Figure \d+|Table \d+|Section \d+', '', text)
    return re.findall(r'[a-z][a-z-]{2,}', text.lower())


def shingles(words, n=8):
    return set(tuple(words[i:i + n]) for i in range(len(words) - n + 1))


def tex_words(path):
    t = open(path, encoding='utf-8', errors='ignore').read()
    t = re.sub(r'(?<!\\)%.*', '', t)
    t = re.sub(r'\\begin\{(equation|table|figure|algorithm)\*?\}.*?\\end\{\1\*?\}', ' ', t, flags=re.S)
    t = re.sub(r'\\bibliography\{[^}]*\}.*', ' ', t, flags=re.S)
    t = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?', ' ', t)
    return t


def resolve(t, d):
    import os
    def rep(m):
        for c in (m.group(1), m.group(1) + '.tex'):
            p = os.path.join(d, c)
            if os.path.exists(p):
                return open(p, encoding='utf-8', errors='ignore').read()
        return ' '
    for _ in range(5):
        nt = re.sub(r'\\input\{([^}]+)\}', rep, t)
        if nt == t:
            break
        t = nt
    return t


# P11: 从PDF取正文(引言到参考文献前)
d11 = fitz.open(r'D:/0科研/工作1/第11篇SCI/paper/DKE_StabilityNotValidity.pdf')
t11 = ''
for p in range(d11.page_count):
    t11 += d11[p].get_text() + '\n'
i = t11.find('Introduction')
t11 = t11[:t11.rfind('References') if 'References' in t11 else len(t11)]
t11 = t11[i:] if i > 0 else t11
w11 = norm_words(t11)
s11 = shingles(w11)
print(f'P11 正文词数 {len(w11)}, 8词shingles {len(s11)}')

# P4/P5/P10
targets = {
    'P4 IJAIT(归属成败)': r'D:/0科研/工作1/第4篇SCI/paper/main_ijait.tex',
    'P5 IJPRAI(预测检测失败)': r'D:/0科研/工作1/第5篇SCI/paper/WorldScientific_IJPRAI_Predicting_Where_Anomaly_Detectors_Fail.tex',
    'P10 Computing(假警报归因)': r'D:/0科研/工作1/第10篇SCI/paper/IDA_SAGE_FalseAlarmAttribution.tex',
}
import os
for name, path in targets.items():
    t = resolve(tex_words(path), os.path.dirname(path))
    w = norm_words(t)
    s = shingles(w)
    overlap = s11 & s
    rate = len(overlap) / len(s11) * 100
    print(f'{name}: 词数{len(w)}, shingle重叠 {len(overlap)} ({rate:.2f}% of P11)')
    if overlap:
        sample = list(overlap)[:3]
        for sh in sample:
            print('   样例: ' + ' '.join(sh))
