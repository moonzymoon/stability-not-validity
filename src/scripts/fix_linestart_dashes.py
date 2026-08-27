# -*- coding: utf-8 -*-
"""Fix the 5 remaining line-start em-dashes with hand-chosen punctuation."""
TEX = r'D:\0科研\工作1\第11篇SCI\paper\DKE_StabilityNotValidity.tex'
t = open(TEX, encoding='utf-8').read()

REPL = [
    ("invariance certificate}, abbreviated \\emph{certificate} throughout\n--- admits a precise statement; we record it because the",
     "invariance certificate}, abbreviated \\emph{certificate} throughout,\nadmits a precise statement; we record it because the"),
    ("(PropRank 0.60 silver / 0.55 PCMCI) GRAA's rows are equal or ahead\n--- the modest gap between the two PropRank PCMCI readings reflects",
     "(PropRank 0.60 silver / 0.55 PCMCI) GRAA's rows are equal or ahead;\nthe modest gap between the two PropRank PCMCI readings reflects"),
    ("anomaly type prevailing in deployment is itself uncertain, which enlarges\n--- rather than shrinks, the space of stable-wrong",
     "anomaly type prevailing in deployment is itself uncertain, which\nenlarges, rather than shrinks, the space of stable-wrong"),
    ("episode label = mean of window labels, thresholded at 0.5) yields 0.91\n--- we report it as a robustness diagnostic, not as the headline,",
     "episode label = mean of window labels, thresholded at 0.5) yields 0.91;\nwe report it as a robustness diagnostic, not as the headline,"),
    ("temporal and exchangeable splits (risk $\\geq$0.44 at $\\alpha{=}0.3$)\n--- driven by cross-dataset heterogeneity.",
     "temporal and exchangeable splits (risk $\\geq$0.44 at $\\alpha{=}0.3$),\ndriven by cross-dataset heterogeneity."),
]
ok = 0
for old, new in REPL:
    if old in t:
        t = t.replace(old, new, 1)
        ok += 1
    else:
        print('NOT FOUND:', old[:70])
open(TEX, 'w', encoding='utf-8').write(t)
print(f'{ok}/5 replaced; remaining --- count: {t.count("---")}')
