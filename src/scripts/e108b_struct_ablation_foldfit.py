# -*- coding: utf-8 -*-
"""E108b: E108 的折内预处理修正版 (回应 预提交审计: 全样本中位数填补与
标准化须在训练折内拟合). 协议其余不变: 轨迹级 LODO logreg, 三特征组,
留一方法(同时剔除训练与测试中的该方法 — 子集复现口径, 论文措辞据此).

数值块在每折内做: 训练折中位数填补 -> 训练折 mu/sd 标准化 -> 应用到测试折.
单热与缺失指示列不经拟合, 原样传递. 输出覆盖 e108_struct_ablation.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
CACHE = os.path.join(SRC, '_cache')

from scripts.e105_stablewrong import (NUM_FEATS, CAT_FEATS, build_matrix,
                                      group_of)


def lodo_auc_foldfit(Xnum, Xfix, y, groups, mask=None):
    """Xnum: 含NaN数值块; Xfix: 单热/指示列(不拟合). 折内预处理."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    m = np.ones(len(y), bool) if mask is None else mask
    aucs = []
    for g in sorted(set(groups[m])):
        tr = m & (groups != g)
        te = m & (groups == g)
        if y[te].std() == 0 or y[tr].std() == 0:
            continue
        med = np.nanmedian(Xnum[tr], axis=0)
        med = np.where(np.isnan(med), 0.0, med)

        def prep(R):
            Z = np.where(np.isnan(R), med, R)
            return Z
        Ztr = prep(Xnum[tr])
        mu, sd = Ztr.mean(axis=0), Ztr.std(axis=0) + 1e-9
        Atr = np.column_stack([(Ztr - mu) / sd, Xfix[tr]])
        Ate = np.column_stack([(prep(Xnum[te]) - mu) / sd, Xfix[te]])
        lr = LogisticRegression(max_iter=4000, C=1.0)
        lr.fit(Atr, y[tr])
        aucs.append(float(roc_auc_score(y[te],
                                        lr.predict_proba(Ate)[:, 1])))
    return float(np.mean(aucs)), len(aucs)


def main():
    e3 = json.load(open(os.path.join(CACHE, 'e3_windows.json'),
                        encoding='utf-8'))
    e9 = json.load(open(os.path.join(CACHE, 'e9_relation_windows.json'),
                        encoding='utf-8'))
    for r in e3 + e9:
        r['_grp'] = group_of(r['dataset'])
    stable = [r for r in e3 + e9 if r['c_cons_topk3'] >= 0.999]
    y = np.array([1 - r['label'] for r in stable])
    groups = np.array([r['_grp'] for r in stable])
    methods = np.array([r['method'] for r in stable])

    Xall, feats = build_matrix(stable)
    n_num = len(NUM_FEATS)
    Xnum = Xall[:, :n_num].astype(float)          # 数值块(含NaN)
    missind = np.isnan(Xnum).any(axis=1).astype(float)[:, None]
    Xoh = Xall[:, n_num:]                          # 单热块(不拟合)
    Xfix = np.column_stack([missind, Xoh])
    print('numeric', Xnum.shape, '| fixed', Xfix.shape,
          '| NaN cells', int(np.isnan(Xnum).sum()))

    out = {}
    a, k = lodo_auc_foldfit(Xnum, Xfix, y, groups)
    out['joint'] = dict(auc=round(a, 4), folds=k)
    a, k = lodo_auc_foldfit(Xnum, np.hstack([missind, np.zeros(
        (len(y), Xoh.shape[1]))]), y, groups)
    out['structure_only'] = dict(auc=round(a, 4), folds=k)
    a, k = lodo_auc_foldfit(np.zeros((len(y), 1)), Xoh, y, groups)
    out['identity_only'] = dict(auc=round(a, 4), folds=k)
    print('joint', out['joint'], '| struct', out['structure_only'],
          '| ident', out['identity_only'])

    out['leave_one_method'] = {}
    for meth in sorted(set(methods)):
        if meth == 'Random':
            continue
        m = methods != meth
        if m.sum() < 300:
            continue
        a, k = lodo_auc_foldfit(Xnum, Xfix, y, groups, mask=m)
        out['leave_one_method'][meth] = dict(auc=round(a, 4), folds=k)
        print(f'  drop {meth:14s} AUC={a:.4f} ({k} folds)')

    out['_protocol'] = ('fold-internal median imputation and z-scoring '
                        '(e108b); leave_one_method drops the method from '
                        'both train and test (subset-replication reading)')
    json.dump(out, open(os.path.join(CACHE, 'e108_struct_ablation.json'),
                        'w'), ensure_ascii=False, indent=1)
    print('written e108_struct_ablation.json (fold-fit protocol)')


if __name__ == '__main__':
    main()
