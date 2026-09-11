# -*- coding: utf-8 -*-
"""轮5整合: AERCA附录表引用 + 6条文献 + 结论包络限定词."""
import re

TEX = r'D:\0科研\工作1\第11篇SCI\paper\DKE_StabilityNotValidity.tex'
BIB = r'D:\0科研\工作1\第11篇SCI\paper\references.bib'

t = open(TEX, encoding='utf-8').read()

# C: AERCA 块加表引用
old = ("Broad grid-wide\nevaluation of published methods remains future work; the point here\n"
       "is that the protocol accepts them without modification and its\n"
       "conclusions are not an artifact of self-implemented probes.")
new = old[:-1] + (" Per-unit values are in\n"
                  "Table~\\ref{tab:app-aerca} (\\cref{app:additional}).")
assert t.count(old) == 1, 'aerca block tail'
t = t.replace(old, new, 1)

# C: 附录 \input
old2 = "\\input{tables/tab_app_gnn}"
assert old2 in t
t = t.replace(old2, "\\input{tables/tab_app_gnn}\n\\input{tables/tab_aerca}", 1)

# D: 文献织入 (6 条)
pairs = [
    ("deployed technology~\\cite{chandola2009survey, su2019omnianomaly,",
     "deployed technology~\\cite{chandola2009survey, goldstein2016survey,\n su2019omnianomaly, hundman2018spacecraft,"),
    ("the anomaly scorer varies in family and training budget;",
     "the anomaly scorer varies in family and training\nbudget~\\cite{breunim2000lof};"),
    ("Closer in spirit is the benchmark-flaw line of Wu and\nKeogh~\\cite{wu2023benchmarks}: our study is the attribution-side",
     "Closer in spirit is the benchmark-flaw line~\\cite{keogh2003need,\n dau2019ucr, wu2023benchmarks}: our study is the attribution-side"),
    ("dependency graph~\\cite{spirtes2000causation, pearl2009causality,\n runge2019pcmci} has imperfect recall and",
     "dependency graph~\\cite{spirtes2000causation, pearl2009causality,\n runge2019inferring, runge2019pcmci} has imperfect recall and"),
]
for o, n in pairs:
    assert t.count(o) == 1, 'weave: ' + o[:40]
    t = t.replace(o, n, 1)

# E: 结论限定词
oldE = "even a formally certified\nstability"
if oldE not in t:
    oldE = "even a formally certified stability"
assert oldE in t, 'conclusion anchor'
t = t.replace(oldE, "even an envelope-conditional certified stability", 1)

open(TEX, 'w', encoding='utf-8').write(t)

bib = open(BIB, encoding='utf-8').read()
adds = r"""
@inproceedings{breunim2000lof,
  author    = {Breunig, Markus M. and Kriegel, Hans-Peter and Ng, Raymond T.
               and Sander, J{\"o}rg},
  title     = {{LOF}: identifying density-based local outliers},
  booktitle = {Proc. ACM SIGMOD},
  year      = {2000}
}

@inproceedings{goldstein2016survey,
  author    = {Goldstein, Markus and Uchida, Seiichi},
  title     = {A comparative evaluation of unsupervised anomaly detection
               algorithms for multivariate data},
  booktitle = {Proc. PAKDD},
  year      = {2016}
}

@inproceedings{hundman2018spacecraft,
  author    = {Hundman, Kyle and Constantinou, Valentino and Laporte,
               Christopher and Colwell, Ian and Soderstrom, Tom},
  title     = {Detecting spacecraft anomalies using {LSTMs} and nonparametric
               dynamic thresholding},
  booktitle = {Proc. KDD},
  year      = {2018}
}

@article{runge2019inferring,
  author  = {Runge, Jakob and Nowack, Peer and Kretschmer, Marlene and
             Flaxman, Seth and Sejdinovic, Dino},
  title   = {Inferring causation from time series in the {Earth} system
             sciences},
  journal = {Earth System Science Data},
  volume  = {11},
  pages   = {1941--1947},
  year    = {2019}
}

@article{keogh2003need,
  author  = {Keogh, Eamonn and Kasetty, Shruti},
  title   = {On the need for time series data mining benchmarks: a survey
             and empirical demonstration},
  journal = {Data Mining and Knowledge Discovery},
  volume  = {7},
  number  = {4},
  pages   = {349--371},
  year    = {2003}
}

@article{dau2019ucr,
  author  = {Dau, Hoang Anh and Bagnall, Anthony and Kamgar, Kaveh and
             Yeh, Chin-Chia Michael and Zhu, Yan and Gharghabi, Shaghayegh
             and Ratanamahatana, Chotirat Ann and Keogh, Eamonn},
  title   = {The {UCR} time series archive},
  journal = {IEEE/CAA Journal of Automatica Sinica},
  volume  = {6},
  number  = {6},
  pages   = {1293--1305},
  year    = {2019}
}
"""
open(BIB, 'w', encoding='utf-8').write(bib + adds)
print('tab_aerca ref + 6 refs + conclusion qualifier done')
