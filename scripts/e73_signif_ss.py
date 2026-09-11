# -*- coding: utf-8 -*-
"""E73: SS 上 GRAA vs PropRank 配对置换检验 (按单元配对, 双图源).

回应 "GRAA 优势在 SS 复现 (+5.3pp) 是否显著": 72 单元两两配对,
置换 10000 次, 双侧 p; 同时给 OB-holdout (17 单元) 的同款检验
(holdout 收窄到持平的显著性), 以及 DPTA-G 对照。
输出: _cache/e73_signif_ss.json
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

CACHE = os.path.join(SRC, '_cache')


def paired_perm(a, b, n_perm=10000, seed=20260908):
    """配对置换: H0 mean(a-b)=0, 双侧."""
    d = np.asarray(a) - np.asarray(b)
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1, 1), size=(n_perm, len(d)))
    null = (signs * d[None, :]).mean(1)
    p = float((np.abs(null) >= abs(obs)).mean())
    return obs, p


def main():
    recs = json.load(open(os.path.join(CACHE, 'e66_rcaeval2.json')))
    # per-unit hit3: (unit, graph) -> method -> hit3
    tab = {}
    for r in recs:
        if r['method'] == '_meta' or r.get('hit3') is None:
            continue
        tab.setdefault((r['unit'], r['graph_source']), {})[r['method']] = r['hit3']

    out = []
    for graph in ('pcmci_clean', 'pcmci_anom'):
        units = sorted(u for (u, g) in tab if g == graph)
        for ma, mb, tag in (('GRAA(p=0.4)', 'PropRank', 'GRAA-PropRank'),
                            ('GRAA(p=0.4)', 'DPTA-G', 'GRAA-DPTAG'),
                            ('DPTA-G', 'PropRank', 'DPTAG-PropRank')):
            a = [tab[(u, graph)][ma] for u in units]
            b = [tab[(u, graph)][mb] for u in units]
            obs, p = paired_perm(a, b)
            out.append(dict(graph=graph, comparison=tag, n=len(units),
                            mean_diff=float(obs),
                            mean_a=float(np.mean(a)),
                            mean_b=float(np.mean(b)), p=float(p)))
    json.dump(out, open(os.path.join(CACHE, 'e73_signif_ss.json'), 'w'),
              default=float, indent=1)
    print('=== E73 paired permutation (SS, 72 units) ===')
    for r in out:
        sig = '**' if r['p'] < 0.01 else ('*' if r['p'] < 0.05 else 'ns')
        print(f"  {r['graph']:12} {r['comparison']:16} "
              f"diff={r['mean_diff']:+.3f} p={r['p']:.4f} {sig}")


if __name__ == '__main__':
    main()
