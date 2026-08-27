# -*- coding: utf-8 -*-
"""统一口径统计第1-11篇论文正文词数（剔除表格/图/公式/参考文献，与 word_count.py 同算法）。"""
import re
import os

BASE = r'D:/0科研/工作1'

TEX = {
    'P2':  r'第2篇SCI/Contrastive_TopK_MIL/paper/IJMLC_Contrastive_TopK_MIL_MTSAD.tex',
    'P3':  r'第3篇SCI/paper/main.tex',
    'P4':  r'第4篇SCI/paper/main_ijait.tex',
    'P5':  r'第5篇SCI/paper/WorldScientific_IJPRAI_Predicting_Where_Anomaly_Detectors_Fail.tex',
    'P6':  r'第6篇SCI/paper/Springer-KAIS-ConformalMTSAD.tex',
    'P8':  r'第8篇SCI/paper/ExpertSystems_CausallyGroundedLLM.tex',
    'P9':  r'第9篇SCI/paper/Elsevier_JLPPI_DETL_合并全文版.tex',
    'P10': r'第10篇SCI/paper/IDA_SAGE_FalseAlarmAttribution.tex',
    'P11': r'第11篇SCI/paper/DKE_StabilityNotValidity.tex',
}
DOCX = {
    'P1': r'第1篇SCI/IEEE_Access_作者交付包_2026-08-03/01_最终稿件/IEEE_Access_Leakage_Resistant_MIL_Final.docx',
}

ENV = r'equation|table|figure|algorithm|verbatim|lstlisting|thebibliography|IEEEkeywords|keywords'


def resolve_inputs(text, dirname, depth=0):
    if depth > 6:
        return text
    def rep(m):
        name = m.group(1)
        for cand in (name, name + '.tex'):
            p = os.path.join(dirname, cand)
            if os.path.exists(p):
                return resolve_inputs(open(p, encoding='utf-8', errors='ignore').read(),
                                      os.path.dirname(p), depth + 1)
        return ' '
    return re.sub(r'\\input\{([^}]+)\}', rep, text)


def tex_words(path):
    t = open(path, encoding='utf-8', errors='ignore').read()
    t = re.sub(r'(?<!\\)%.*', '', t)
    t = resolve_inputs(t, os.path.dirname(path))
    i = t.find(r'\begin{document}')
    if i >= 0:
        t = t[i:]
    t = re.sub(r'\\begin\{(' + ENV + r')\*?\}.*?\\end\{\1\*?\}', ' ', t, flags=re.S)
    t = re.sub(r'\\bibliography\{[^}]*\}.*', ' ', t, flags=re.S)
    t = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?', ' ', t)
    t = re.sub(r'[{}&$~^_\\]', ' ', t)
    return len([w for w in t.split() if re.search(r'[A-Za-z]', w)])


def docx_words(path):
    from docx import Document
    doc = Document(path)
    texts = [p.text for p in doc.paragraphs]
    joined = '\n'.join(texts)
    m = re.search(r'^\s*(References|REFERENCES|Reference)\s*$', joined, flags=re.M)
    if m:
        joined = joined[:m.start()]
    return len([w for w in joined.split() if re.search(r'[A-Za-z]', w)])


print(f"{'篇':4s} {'正文词数':>8s}  文件")
print('-' * 70)
results = {}
for pid, rel in sorted(TEX.items()):
    p = os.path.join(BASE, rel)
    try:
        n = tex_words(p)
        results[pid] = n
        print(f"{pid:4s} {n:8d}  {os.path.basename(rel)}")
    except Exception as e:
        print(f"{pid:4s}     FAIL  {rel}: {e}")
for pid, rel in DOCX.items():
    p = os.path.join(BASE, rel)
    try:
        n = docx_words(p)
        results[pid] = n
        print(f"{pid:4s} {n:8d}  {os.path.basename(rel)} (docx,含图注以外的全部段落)")
    except Exception as e:
        print(f"{pid:4s}     FAIL  {rel}: {e}")

vals = [v for v in results.values() if v]
import statistics
print('-' * 70)
print(f"已投稿正文词数: 中位数={statistics.median(vals):.0f}  "
      f"均值={statistics.mean(vals):.0f}  最少={min(vals)}  最多={max(vals)}")
