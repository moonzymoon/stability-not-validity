# -*- coding: utf-8 -*-
"""同步第11篇(DKE)信息到台账 row13. 不触碰凭据列之外内容."""
import openpyxl

P = r'D:\0科研\工作1\已投稿\已投稿论文汇总表.xlsx'
wb = openpyxl.load_workbook(P)
ws = wb['已投稿论文汇总']

row = 13  # 第十一篇
updates = {
    1: None,                                   # 进度: 未投
    2: 'Elsevier',
    3: 'Data & Knowledge Engineering',
    4: 'DKE',
    5: '中科院3区 / JCR Q2（投稿当月LetPub复核）',
    6: '3.9',
    7: ('Stability Is Not Validity: A Reliability Benchmark for '
        'Root-Cause Attribution in Multivariate Time-Series Anomaly '
        'Detection'),
    8: 'Yao Zhang, Yanling Li(通讯), Baocai Li',
    9: None,                                   # Username 待注册EM
    10: 'liyanling@bdu.edu.cn',
    12: ('2026-08-26 定稿待投稿：71页review格式/19表6图2算法/40文献，'
         'DKE指南全合规+AERCA(ICLR2025)外部基线双协议+约90条外部审稿'
         '意见闭环；拒稿预估~13-15%。投稿前待办：Vitae填研究方向、'
         'CRediT分工与共同作者确认、EM官方Declaration of Interests表、'
         '当月LetPub复核；4个月未送审检查点→转投NPL'),
    13: None,                                  # Submission ID
}
old = {c: ws.cell(row=row, column=c + 1).value for c in updates}
for c, v in updates.items():
    ws.cell(row=row, column=c + 1).value = v
wb.save(P)

safe_old = {c: old[c] for c in (1, 2, 3, 4, 5, 6, 7, 8, 12)}
print('改动前(安全列):')
for c, v in safe_old.items():
    print(' ', c, str(v)[:60])
print('已写入 row13 第十一篇 = DKE / 李燕玲通讯 / 2026-08-26定稿待投')
