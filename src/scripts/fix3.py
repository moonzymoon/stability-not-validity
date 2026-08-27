# -*- coding: utf-8 -*-
"""收敛轮剩余表编辑: LODSO range + SWaT ACR 原因."""
edits = [
    ('tab_predictor.tex',
     '(LODSO, 48 held-out combinations) reaches 0.80.',
     '(LODSO, 48 held-out combinations) reaches 0.80 (range '
     '0.60--1.00; full model).'),
    ('tab_swat.tex',
     'Matched isolation-forest run (for the GRAA comparison), '
     'hit@1/hit@3 only.}',
     'Matched isolation-forest run (for the GRAA comparison), '
     'hit@1/hit@3 only; hit@5 and ACR were not computed in this '
     'source-swap-focused run.}'),
]
for f, old, new in edits:
    t = open(f, encoding='utf-8').read()
    assert old in t, (f, old[:50])
    open(f, 'w', encoding='utf-8').write(t.replace(old, new))
    print('patched', f)
