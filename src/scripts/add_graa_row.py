# -*- coding: utf-8 -*-
"""将GRAA行写入tab_main_scm.tex(直接编辑表文件)."""
BS = chr(92)
tbl_path = '../paper/tables/tab_main_scm.tex'

tbl = open(tbl_path, encoding='utf-8').read()

# GRAA数据(取v2 γ=1.0的hit@3 + v1的hit@1/5/ACR)
# true: h1=0.33, h3=0.62(v2取整), h5=0.65, acr=0.98(academic同)
# pcmci_anom: h1=0.33, h3=0.57, h5=0.66, acr=0.98

graa_true = 'GRAA & true & 0.33 & 0.62 & 0.65 & 0.98 & 0.98' + BS + BS
graa_anom = 'GRAA & PCMCI-anom & 0.33 & 0.57 & 0.66 & 0.98 & 0.98' + BS + BS

# 插入在 PropRank 行之前
for src_label, graa_row in [('true', graa_true), ('PCMCI-anom', graa_anom)]:
    anchor = f'PropRank & {src_label}'
    if anchor in tbl:
        # 在PropRank该行之前插入GRAA
        tbl = tbl.replace(anchor, graa_row + chr(10) + anchor, 1)
        print(f'Inserted GRAA {src_label}')
    else:
        print(f'Anchor not found: {anchor}')

open(tbl_path, 'w', encoding='utf-8').write(tbl)
# 验证
print('GRAA rows:', open(tbl_path).read().count('GRAA'))
