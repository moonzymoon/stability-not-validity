# -*- coding: utf-8 -*-
"""E86: 预测器模型类对比 (LR / RF / HGB / MLP), LODO on e3 SCM 窗口特征.
回答"为什么 HGB"。输出 e86_model_compare.json"""
import json
import os

import numpy as np
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '..', '_cache')

rows = json.load(open(os.path.join(CACHE, 'e3_windows.json'),
                      encoding='utf-8'))
feats = [c for c in rows[0] if len(c) > 2 and c[0] in 'gcasd'
         and c[1] == '_']
datasets = sorted(set(r['dataset'] for r in rows))
X = np.array([[r.get(f, np.nan) for f in feats] for r in rows])
y = np.array([r['label'] for r in rows])
dcol = np.array([r['dataset'] for r in rows])
print('rows', len(rows), 'feats', len(feats), 'datasets', datasets)

models = {
    'logistic': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                              LogisticRegression(max_iter=2000,
                                                 random_state=0)),
    'randomforest': make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(n_estimators=300,
                                           random_state=0, n_jobs=-1)),
    'hgb': HistGradientBoostingClassifier(random_state=0),
    'mlp': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=(64, 32),
                                       max_iter=400, random_state=0)),
}
out = {}
for name, clf in models.items():
    aucs = []
    for d in datasets:
        tr, te = dcol != d, dcol == d
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        if len(set(y[te])) > 1:
            aucs.append(roc_auc_score(y[te], p))
    out[name] = dict(lodo_mean=float(np.mean(aucs)),
                     lodo_per_fold=[round(float(a), 3) for a in aucs])
    print(name, out[name], flush=True)

json.dump(out, open(os.path.join(CACHE, 'e86_model_compare.json'), 'w'),
          indent=1)
print('-> e86_model_compare.json')
