# -*- coding: utf-8 -*-
"""E90: R3 覆写规则的量化落地 (#6 可计算部分).

R3: 弱异常档 x 替换式归因 x 高一致性 -> 强制人审.
从 e3_windows.json (SCM 全网格窗口特征) 统计:
  触发率, 触发组内错误率 vs 未触发组错误率, 提升倍数
-> _cache/e90_r3_quant.json
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '..', '_cache')

rows = json.load(open(os.path.join(CACHE, 'e3_windows.json'),
                      encoding='utf-8'))
REPLACEMENT = {'GlobalCF', 'CondAttr'}
WEAK = '_m0.25'

trig, non = [], []
for r in rows:
    m = r.get('method', '')
    if m in REPLACEMENT and WEAK in r.get('dataset', '') \
            and r.get('c_cons_topk3', 0) > 0.8:
        trig.append(r['label'])
    else:
        non.append(r['label'])

trig = np.array(trig)
non = np.array(non)
out = dict(
    n_trigger=int(len(trig)), n_total=int(len(trig) + len(non)),
    trigger_rate=float(len(trig) / (len(trig) + len(non))),
    wrong_given_trigger=float(1 - trig.mean()) if len(trig) else None,
    wrong_given_rest=float(1 - non.mean()),
    lift=(float((1 - trig.mean()) / (1 - non.mean()))
          if len(trig) and non.mean() < 1 else None),
    base_wrong=float(1 - np.concatenate([trig, non]).mean()),
)
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(CACHE, 'e90_r3_quant.json'), 'w'),
          indent=1)
print('-> e90_r3_quant.json')
