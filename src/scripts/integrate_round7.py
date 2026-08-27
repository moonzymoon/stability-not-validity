# -*- coding: utf-8 -*-
"""轮7: 三外审综合修复 F2-F25 (F1/F26/F27 待实验结果)."""
t = open(r'D:\0科研\工作1\第11篇SCI\paper\DKE_StabilityNotValidity.tex',
         encoding='utf-8').read()
n = [0]

def rep(old, new, tag):
    global t
    assert t.count(old) == 1, 'MISS ' + tag
    t = t.replace(old, new, 1)
    n[0] += 1

# F2 prevalence 措辞
rep("Brier\nscore 0.158 against a prevalence baseline of 0.226",
    "Brier\nscore 0.158 against the no-skill baseline of 0.226 (the\nprevalence-0.34 coin flip)", 'F2')

# F3 17400 构成
rep("$n{=}17{,}400$\nout-of-fold windows",
    "$n{=}17{,}400$\nout-of-fold window-level samples (nine methods pooled)", 'F3')

# F4a TE 0.34 定位
rep("but \\emph{loses} $16$ points on TE over the same\nbudget sweep ($0.34\\to0.18$);",
    "but \\emph{loses} $16$ points on TE over the same\nbudget sweep ($0.34\\to0.18$; PCA-only quarter-budget cells,\nTable~\\ref{tab:app-scorer}, outside the main grid whose\nfull-budget three-scorer TE maximum is 0.25);", 'F4a')
# F4b 主结果句加限定
rep("On TE (Table~\\ref{tab:main-te}) no method exceeds hit@3 0.25",
    "On TE (Table~\\ref{tab:main-te}) no method exceeds hit@3 0.25 in\nthe full-budget three-scorer grid", 'F4b')

# F5 beta 定义
rep("the mixing exponent $\\beta$, which reshapes exactly this\naggregation, moves results by at most 2pp",
    "the mixing exponent $\\beta$ (the gate is raised to the power\n$\\beta$, i.e.\\ $w_{\\mathrm{prop}}{=}\\bar{c}^{\\,\\beta}$; $\\beta{=}1$\nthroughout), which reshapes exactly this aggregation, moves\nresults by at most 2pp", 'F5')

# F6 LODO 首现展开
rep("Validation:\nLODO and the leak-free LODSO protocol",
    "Validation:\nleave-one-dataset-out (LODO) and the leak-free\nleave-one-(dataset$\\times$scorer)-out (LODSO) protocol", 'F6')

# F7 AERCA 扰动协议说明
rep("\\emph{Stability}: under input-noise perturbation\nof the scored windows (5--10\\% of channel standard deviation), its",
    "\\emph{Stability}: AERCA's graph is learned internally and never\nexposed as an input, so its stability axis is input-noise\nperturbation of the scored windows (5--10\\% of channel standard\ndeviation, a band chosen for its own dynamic range rather than\nthe main grid's MAD-calibrated one; its ACR figures are\nself-contained and are not pooled into the grid's cross-family\ncomparisons), and its", 'F7')

# F8 Cor2 软化
rep("exactly as Corollary~\\ref{cor:cov} predicts for a higher-$\\eta$\nregime:",
    "consistent with the envelope-size ordering of\nCorollary~\\ref{cor:cov}, which strictly compares configurations\nsharing margins and is read here as an empirical regularity\nrather than a prediction:", 'F8')

# F9 结论 36% 限定
rep("even an envelope-conditional certified stability\n(Proposition~\\ref{prop:cert}) leaves 36\\% of certified windows wrong.",
    "even an envelope-conditional certified\nstability (Proposition~\\ref{prop:cert}) leaves 36\\% of the\ncertified windows in our benchmark wrong.", 'F9')

# F10 sustained
rep("which variables\nsustained an alarm", "which variables\ncaused an alarm", 'F10')

# F11 absolute
rep("shows\ngroup means within 0.04 of each other",
    "shows\ngroup means within an absolute 0.04 of each other", 'F11')

# F12 K=3
rep("is $K/d$\nfor single-root scenes (0.20 on the 15-variable SCM",
    "is $K/d$\nfor single-root scenes (at $K{=}3$: 0.20 on the 15-variable SCM", 'F12')

# F13 RBO 取消声明弱化
rep("so it shifts all RBO values\nequally and cancels in comparisons (RBO at $p{=}0.9$ is insensitive to\npermutations beyond the first ranks).",
    "so its contribution is identical only for methods that agree on\nthat order; at $p{=}0.9$ the weighting concentrates on the top\nranks, and we read RBO differences as driven by the cause-ranked\nhead, using RBO only as a secondary within-method diagnostic.", 'F13')

# F14 u_m 恒等排除
rep("the inner expectation draws seeded perturbation instances ($M{=}8$\nper operator/strength)",
    "the inner expectation draws seeded perturbation instances ($M{=}8$\nper operator/strength; instances never coincide with $u_0$)", 'F14')

# F15 floor
rep("drop the lowest-confidence\n$p\\,|E|$ edges",
    "drop the lowest-confidence\n$\\lfloor p\\,|E|\\rfloor$ edges", 'F15')

# F16 ECE 分箱
rep("expected calibration error 0.027",
    "expected calibration error 0.027 (ten equal-width bins)", 'F16')

# F17 M_c 措辞
rep("\\emph{perturbation consistency} of the window (top-$K$ overlap under\n$M_c{=}6$ feature perturbations, $K{=}3$",
    "\\emph{perturbation consistency} of the window (top-$K$ overlap under\n$M_c{=}6$ perturbations of the method's own uncertainty input,\ngraph operators for graph-dependent methods and input\nnoise/misalignment for graph-free ones, $K{=}3$", 'F17')

# F18 弱异常句软化
rep("precisely the regime where operators need attribution most",
    "a regime of practical concern: mature systems engineer away\ngross faults, so the surviving alarm population is\ndisproportionately weak", 'F18')

# F19 污染场景现实性
rep("\\emph{contamination}, PCMCI on\nanomaly-contaminated data (precision 0.22--0.67) costs PropRank 7.0",
    "\\emph{contamination}, PCMCI on\nanomaly-contaminated data (precision 0.22--0.67; not a contrived\nchoice: when clean normal-period data is scarce or stale, learning\nmust use recent mixed data, and the RCAEval construction learns\non the full deployed segment) costs PropRank 7.0", 'F19')

# F20 机理-参考系对应
rep("The quadrant is populated by (at least) three\nmechanisms",
    "The quadrant is populated by (at least) three\nmechanisms (the first two are observable under the academic\nreference with the true graph; the third requires the deployment\nreference, where the reference itself is biased)", 'F20')

# F21 象限跨族说明
rep("(Table~\\ref{tab:quadrant}, visualized in Figure~\\ref{fig:quadrant});",
    "(Table~\\ref{tab:quadrant}, visualized in\nFigure~\\ref{fig:quadrant}; each configuration is scored against\nits own family's declared perturbation band, so the share\naggregates the declared protocols rather than mixing scales);", 'F21')

# F22 GRAA 自身 ACR
rep("Between graph sources GRAA varies by only 2.0pp (true 0.609,",
    "GRAA's own ACR@3 is 0.735--0.741 across graph sources: it\nremains a transmitting consumer whose stability tracks its\ninput quality rather than certifying correctness, exactly the\nbehavior the benchmark asks of an honest graph method.\nBetween graph sources GRAA's hit@3 varies by only 2.0pp (true 0.609,", 'F22')

# F23 RCAEval 0.99 说明
rep("the\nreconstruction (0.99) and deviation (0.78) readers dominate while",
    "the\nreconstruction (0.99) and deviation (0.78) readers dominate (the\nfault signature spans 4--5 jointly-deviating metrics of one\nservice, so readers that flag joint deviation approach ceiling;\nthe Random control's 0.24 calibrates the scale) while", 'F23')

# F24 conformal 软化
rep("driven by cross-dataset heterogeneity.",
    "plausibly driven by cross-dataset heterogeneity.", 'F24')

# F25 P 符号重载
rep("\\mathcal{P}_{\\mathrm{del}}/\\mathcal{P}_{\\mathrm{add}}/\n\\mathcal{P}_{\\mathrm{rew}}$ for graph-dependent methods",
    "\\mathcal{Q}_{\\mathrm{del}}/\\mathcal{Q}_{\\mathrm{add}}/\n\\mathcal{Q}_{\\mathrm{rew}}$ (operator sub-families of\n$\\mathcal{Q}_\\delta$) for graph-dependent methods", 'F25')

open(r'D:\0科研\工作1\第11篇SCI\paper\DKE_StabilityNotValidity.tex', 'w',
     encoding='utf-8').write(t)
print(f'{n[0]} fixes applied')
