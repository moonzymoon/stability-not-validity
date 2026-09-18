# -*- coding: utf-8 -*-
"""e52 (purged CV) + e53 (adaptive p) 入文."""
t = open(r'D:\0科研\工作1\第11篇SCI\paper\INS_StabilityNotValidity.tex',
         encoding='utf-8').read()

old1 = "($\\sigma{\\approx}0.02$--$0.04$); the mean is retained throughout."
new1 = old1 + (" An observable adaptive rule for the prune fraction\n"
               "itself, $p^{*}{=}\\mathrm{clip}(1-\\text{mean edge\n"
               "confidence}, 0.1, 0.6)$, which prunes harder when the\n"
               "deployed graph's confidences are low, lands within\n"
               "$\\pm1.7$pp of the fixed $p{=}0.4$ on all three graph\n"
               "sources (0.623/0.626/0.626 vs.\\ 0.629/0.631/0.609), so\n"
               "the fixed default is not leaving performance on the table.")
assert t.count(old1) == 1, 'old1'
t = t.replace(old1, new1, 1)

old2 = "with mean AUROC 0.79 under leave-one-dataset-out"
idx = t.find(old2)
assert idx > 0, 'old2'
end = t.find('. ', idx)
ins = (" A purged temporal split (train on the first 70\\% of each\n"
       "trajectory, test on the final 30\\%, five-window purge gap,\n"
       "datasets pooled as in LODO) reaches AUROC 0.81, so the LODO\n"
       "estimate is not inflated by adjacent-window correlation.")
t = t[:end] + ins + t[end:]
open(r'D:\0科研\工作1\第11篇SCI\paper\INS_StabilityNotValidity.tex', 'w',
     encoding='utf-8').write(t)
print('e52+e53 integrated')
