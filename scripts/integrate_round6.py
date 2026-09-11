# -*- coding: utf-8 -*-
"""轮6: 三外审综合修复 (16项文本修复)."""
t = open(r'D:\0科研\工作1\第11篇SCI\paper\DKE_StabilityNotValidity.tex',
         encoding='utf-8').read()
n_fix = 0

def rep(old, new, tag):
    global t, n_fix
    assert t.count(old) == 1, 'MISS ' + tag
    t = t.replace(old, new, 1)
    n_fix += 1

# 1. 摘要: gain 加十种子衰减 + 压缩删边句保词数
rep("A refined nine-point sweep locates the\nonset of validity loss under edge deletion near 20--25\\%, and the",
    "Validity loss under edge\ndeletion begins near 20--25\\%, and the", 'abstract-trim')
rep("GRAA, a graph-reliability-aware\nattribution method, gains 9.4pp on contaminated graphs at a\nsmall clean-graph cost ($-1.1$pp on the pre-registered\nseeds; $-4.4$pp on a ten-seed extension).",
    "GRAA, a graph-reliability-aware\nattribution method, gains 9.4pp on contaminated graphs on the\npre-registered seeds (+4.9pp on a ten-seed extension), at a\nclean-graph cost of $-4.4$pp.", 'abstract-gain')
rep("Even\ncertified stability should therefore not be read as correctness\nevidence.",
    "Consequently,\nenvelope-certified stability does not constitute evidence of\ncorrectness.", 'abstract-final')

# 2. 象限双口径 (正文首现处)
rep("configurations fill a quarter of the grid (36\\% under weak",
    "configurations fill a quarter of the grid counting the random\ncontrol, 21.7\\% excluding it (the control is maximally stable-wrong\nby construction; 25.2\\% including it), and 36\\% under weak",
    'quadrant-dual')

# 3. Limitations: AERCA 三床
rep("AERCA (official implementation and\nbudget) enters on all four testbeds with both metrics",
    "AERCA (official implementation and\nbudget) enters on three of the four testbeds (RCAEval, SWaT, SCM)\nwith both metrics", 'lim-aerca')

# 4. source-swap 残句
rep("(\\cref{fig:sourceswap} visualizes the transmission pattern by\nmethod and graph source).\nmethod stays high, 0.86",
    "(\\cref{fig:sourceswap} visualizes the transmission pattern by\nmethod and graph source).\nThe transmitting method's deployment-referenced ACR@3 stays high\nat 0.86", 'swap-fix')

# 5. TE 幅度换算
rep("amplitude ratio $\\in \\{0.5, 1, 3\\}$ = weak/medium/strong",
    "amplitude ratio $\\in \\{0.5, 1, 3\\}$ = weak/medium/strong\n(injection magnitudes 0.25/0.5/1.5 at noise std 0.5, the labeling\nused in the difficulty-tier table)", 'te-mag')

# 6. e30 scorer 无关 + Table1 dagger 语义
rep("Table~\\ref{tab:main-scm} includes GRAA ($p{=}0.4$) alongside the\nbaselines.",
    "Table~\\ref{tab:main-scm} includes GRAA ($p{=}0.4$) alongside the\nbaselines. Its rows carry a dagger because they are single-scorer\n(isolation-forest) runs, but this is not a weaker protocol: run\nunder all three scorers, GRAA's hit@3 agrees to the third decimal\n(0.570/0.570/0.570 on the contaminated source, 0.584 on true), as\nits construction consumes only deviation statistics and the graph.",
    'scorer-indep')

# 7. e56 拆解消融 (接在自适应 p 句后)
rep("the fixed default is not leaving performance on the table.",
    "the fixed default is not leaving performance on the table. A\ndissection ablation on the contaminated source separates the two\ncomponents: confidence pruning alone (no ensemble) reaches 0.614,\na fixed-weight ensemble without pruning 0.554, random pruning at\nthe same edge budget 0.511, and the full design 0.629; on the true\ngraph, pruning alone matches the full method (0.609). The gain\ncomes from the confidence signal itself (random pruning is no\nbetter than pruning nothing at all would be, and slightly worse\nthan the unpruned baseline), with the adaptive ensemble adding\n1.5 points only when the graph is contaminated.",
    'dissect')

# 8. 9.4pp Δ 澄清
rep("graphs GRAA gains \\textbf{9.4pp} over pure PropRank (0.629 vs",
    "graphs GRAA gains \\textbf{9.4pp} over pure PropRank ($\\Delta{=}0.094$\non unrounded means; 0.629 vs", 'delta-clar')

# 9. 结论 uninformative signal
rep("together replace an uninformative signal with a usable one: an",
    "together replace perturbation consistency alone with a usable\nsignal: an", 'concl')

# 10. To our knowledge 软化
rep("No prior benchmark separates the two\nproperties measurably",
    "To our knowledge, no prior benchmark separates the two\nproperties measurably", 'tok')

# 11. Adebayo cite
rep("That stable methods can ignore\nthe perturbed input is known;",
    "That stable methods can ignore\nthe perturbed input is known~\\cite{adebayo2018sanity};", 'adebayo')

# 12. AERCA 官方超参 + GRAA 对比
rep("Broad grid-wide\nevaluation of published methods remains future work;",
    "AERCA ran with its official hyperparameters and early stopping,\nuntuned for our testbeds; on the same SCM trajectories GRAA\n(0.58--0.63) is at or above it (0.58). Broad grid-wide\nevaluation of published methods remains future work;", 'aerca-note')

# 13. predictor RCAEval 边界
rep("predictor does not transfer across anomaly types without",
    "predictor is not yet validated on RCAEval windows, and does not\ntransfer across anomaly types without",
    'pred-rca')

# 15. p* 规则边集说明
rep("$p^{*}{=}\\mathrm{clip}(1-\\text{mean edge\nconfidence}, 0.1, 0.6)$",
    "$p^{*}{=}\\mathrm{clip}(1-\\text{mean edge\nconfidence}, 0.1, 0.6)$ (the mean is taken over all learned edges\nbefore pruning)", 'pstar')

# 17. fig_quadrant 阈值注
rep("The stable-wrong\nquadrant (upper left of the crossing) is densely populated.}",
    "The stable-wrong\nquadrant (upper left of the crossing; thresholds ACR@3 $>0.8$ and\nhit@3 $<0.4$) is densely populated.}", 'fig1')

open(r'D:\0科研\工作1\第11篇SCI\paper\DKE_StabilityNotValidity.tex', 'w',
     encoding='utf-8').write(t)
print(f'{n_fix} fixes applied')
