# -*- coding: utf-8 -*-
"""将GRAA行加入tab_main_scm.tex."""
import json, collections
import numpy as np

# 读取GRAA数据
graa = json.load(open('_cache/e18b_graa_v2.json'))
# 读取现有表
tbl_path = '../paper/tables/tab_main_scm.tex'
tbl = open(tbl_path, encoding='utf-8').read()

# 提取GRAA数字(γ=1.0, 5 seeds平均)
agg = collections.defaultdict(list)
for r in graa:
    if r['method'] == 'GRAA-v2(g=1.0)':
        agg[r['graph_source']].append(r['hit3'])

h_true = np.mean(agg['true'])
h_anom = np.mean(agg['pcmci_anom'])
print(f'GRAA true: {h_true:.3f}, anom: {h_anom:.3f}')

# GRAA的hit@1和hit@5 (从e18数据)
graa_v1 = json.load(open('_cache/e18_graa.json'))
agg1 = collections.defaultdict(lambda: dict(h1=[], h3=[], h5=[], a=[]))
for r in graa_v1:
    if r['method'] == 'GRAA':
        k = r['graph_source']
        agg1[k]['h1'].append(r['hit1'])
        agg1[k]['h3'].append(r['hit3'])
        agg1[k]['h5'].append(r['hit5'])
        if r.get('acr3') is not None:
            agg1[k]['a'].append(r['acr3'])

BS = chr(92)
# 在PropRank(true)行之前插入GRAA行
for src, label in [('true', 'true'), ('pcmci_anom', 'PCMCI-anom')]:
    h1 = np.mean(agg1[src]['h1'])
    h3 = np.mean(agg1[src]['h3'])
    h5 = np.mean(agg1[src]['h5'])
    a = np.mean(agg1[src]['a']) if agg1[src]['a'] else None
    a_str = f'{a:.2f}' if a else '--'

    graa_row = f'GRAA & {label} & {h1:.2f} & {h3:.2f} & {h5:.2f} & {a_str} & {a_str}' + BS + BS

    if src == 'true':
        # 插入在PropRank true行之前
        anchor = f'PropRank & true'
        if anchor in tbl:
            tbl = tbl.replace(anchor, graa_row + chr(10) + anchor, 1)
    elif src == 'pcmci_anom':
        anchor = f'PropRank & PCMCI-anom'
        if anchor in tbl:
            tbl = tbl.replace(anchor, graa_row + chr(10) + anchor, 1)

# 保存
open(tbl_path, 'w', encoding='utf-8').write(tbl)
print('GRAA rows added to tab_main_scm.tex')
