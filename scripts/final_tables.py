# -*- coding: utf-8 -*-
"""收敛轮表格修复: dagger/TE CI/LODSO range/SWaT 注."""
BS = chr(92)
D = BS + '$^' + BS + 'dagger' + BS + '$'

edits = [
    # 4: Table1 GRAA 行加剑号 + caption 说明
    ('tab_main_scm.tex', 'GRAA & true &', 'GRAA' + D + ' & true &'),
    ('tab_main_scm.tex', 'GRAA & PCMCI-clean &',
     'GRAA' + D + ' & PCMCI-clean &'),
    ('tab_main_scm.tex', 'GRAA & PCMCI-anom &',
     'GRAA' + D + ' & PCMCI-anom &'),
    ('tab_main_scm.tex',
     'GRAA rows carry seed-level CIs over five trajectories under the '
     'isolation-forest scorer',
     BS + 'dag: isolation-forest scorer only; text comparisons use '
     'matched-scorer baselines. GRAA rows carry seed-level CIs over '
     'five trajectories'),
    # 6: TE GRAA 行加 6 单元 bootstrap CI + caption 措辞
    ('tab_main_te.tex',
     'GRAA(' + BS + '$p{=}0.4' + BS + '$) & PCMCI & 0.00 & 0.25 & -- '
     '& -- & --' + BS + BS,
     'GRAA(' + BS + '$p{=}0.4' + BS + '$) & PCMCI & 0.00 & 0.25 '
     '[.008,.484] & -- & -- & --' + BS + BS),
    ('tab_main_te.tex',
     '(single-scorer post hoc run, hence no CI; its source-swap '
     'behavior is analyzed',
     '(single-scorer run; the bracketed CI is a six-unit bootstrap '
     'and is correspondingly wide; its source-swap behavior is '
     'analyzed'),
    # 7: LODSO range
    ('tab_predictor.tex',
     '(LODSO, 48 held-out combinations) reaches 0.80.',
     '(LODSO, 48 held-out combinations) reaches 0.80 (range '
     '0.60--1.00; full model).'),
    # 8: SWaT GRAA 未算 ACR 原因
    ('tab_swat.tex',
     'Matched isolation-forest run (for the GRAA comparison), '
     'hit@1/hit@3 only.}',
     'Matched isolation-forest run (for the GRAA comparison), '
     'hit@1/hit@3 only; hit@5 and ACR were not computed in this '
     'source-swap-focused run.}'),
]
for f, old, new in edits:
    t = open(f, encoding='utf-8').read()
    assert old in t, (f, old[:60])
    open(f, 'w', encoding='utf-8').write(t.replace(old, new))
    print('patched', f)
