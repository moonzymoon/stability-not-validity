# -*- coding: utf-8 -*-
"""提取块1(引言+相关工作)与块2(方法部分) —— 原文逐字, 不改写."""
t = open('paper/DKE_StabilityNotValidity.tex', encoding='utf-8').read()

i1 = t.find('\\section{Introduction}')
i2 = t.find('\\section{Benchmark design}')
i3 = t.find('\\section{Experiments}')
assert 0 < i1 < i2 < i3, (i1, i2, i3)

b1 = t[i1:i2].strip()
b2 = t[i2:i3].strip()

out = []
out.append('=' * 30 + ' 块1：引言+相关工作 ' + '=' * 30)
out.append(b1)
out.append('\n' + '=' * 30 + ' 块2：方法部分(框架/问题定义) ' + '=' * 30)
out.append(b2)
open('blocks12.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('块1字符:', len(b1), '块2字符:', len(b2))
