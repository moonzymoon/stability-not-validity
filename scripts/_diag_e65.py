# -*- coding: utf-8 -*-
"""诊断: AttNN 代理学不动 iforest 分数 — 任务问题还是架构问题?"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
import sys
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from data.scm import make_sample, build_windows
from scorers import make_scorer
from attribution.attnn import AttnSurrogate

torch.manual_seed(20260906)
torch.set_num_threads(2)

sample = make_sample(n_nodes=15, T=5000, n_single=5, n_joint=2,
                     seed=0, noise_std=0.5, magnitude=0.5)
win = build_windows(sample, window=16, stride=20)
X = win['X_anom'].astype(np.float32)
scorer = make_scorer('iforest').fit(win['X_pool'])
s = scorer.score(X).astype(np.float64)
print('target stats: min=%.3f p50=%.3f max=%.3f std=%.3f' %
      (s.min(), np.median(s), s.max(), s.std()))

rng = np.random.default_rng(0)
idx = rng.permutation(len(X))
va, tr = idx[:20], idx[20:]
mu, sd = s[tr].mean(), s[tr].std() + 1e-9
yn = (s - mu) / sd
const = float((yn[va] ** 2).mean())
print('const val mse =', round(const, 4))


def train(net, Xf, epochs=300, lr=1e-3, tag=''):
    opt = torch.optim.Adam(net.parameters(), lr)
    trX = torch.from_numpy(Xf[tr]).float()
    trY = torch.from_numpy(yn[tr]).float()
    vaX = torch.from_numpy(Xf[va]).float()
    best = np.inf
    for ep in range(epochs):
        opt.zero_grad()
        out = net(trX)
        if isinstance(out, tuple):
            out = out[0]
        loss = nn.functional.mse_loss(out, trY)
        loss.backward()
        opt.step()
        with torch.no_grad():
            vo = net(vaX)
            if isinstance(vo, tuple):
                vo = vo[0]
            v = float(nn.functional.mse_loss(vo,
                                             torch.from_numpy(yn[va]).float()))
        best = min(best, v)
    with torch.no_grad():
        vo = net(vaX)
        if isinstance(vo, tuple):
            vo = vo[0]
        pred = vo.squeeze(-1).numpy()
    corr = float(np.corrcoef(pred, yn[va])[0, 1])
    print(f'{tag:12} best val={best:.4f} ratio={best/const:.3f} corr={corr:.3f}')


train(nn.Sequential(nn.Flatten(), nn.Linear(240, 64), nn.ReLU(),
                    nn.Linear(64, 1)), X, tag='MLP-raw')
s2 = s.copy()
Xn = X.copy()
for j in range(X.shape[2]):
    m = np.median(X[:, :, j])
    mad = np.median(np.abs(X[:, :, j] - m)) * 1.4826 + 1e-9
    Xn[:, :, j] = (X[:, :, j] - m) / mad
train(nn.Sequential(nn.Flatten(), nn.Linear(240, 64), nn.ReLU(),
                    nn.Linear(64, 1)), Xn, tag='MLP-madnorm')
train(AttnSurrogate(), Xn, tag='Attn-madnorm')
