# -*- coding: utf-8 -*-
"""e50 官方满协议 AERCA 双指标结果重写入文 + Limitations/CL 同步."""
t = open(r'D:\0科研\工作1\第11篇SCI\paper\DKE_StabilityNotValidity.tex',
         encoding='utf-8').read()

old = """To check that the benchmark is not self-referential, we ran
AERCA~\\cite{han2025aerca}, a published Granger-causal attribution
method, end-to-end with its official implementation on the two real
injected testbeds, scoring its encoder-latent deviations under our
window protocol (MSDS-style configuration, training budget capped
at 400 epochs with the official early stopping, 5--16 CPU-minutes
per unit; three RCAEval fault cases and the full SWaT injected
unit, 450 windows in total). The pattern of results is itself a
finding. On SWaT, AERCA lands mid-pack (hit@3 0.63, between the
propagation readers and the deviation readers). On RCAEval it is
bimodal across fault cases, 1.00 / 0.00 / 0.01: for one fault
signature its latent deviation ranking is perfect, for two it fails
almost completely, and in both regimes it emits a full, apparently
confident ranking. A state-of-the-art trained method thus shows
exactly the condition-dependent validity that the grid formalizes;
averaged over its three cases (0.34) it sits below the simplest
deviation reader on the same testbed (0.78), while its best case
tops every family. Broad evaluation of published end-to-end methods
across the full grid remains future work; the point here is that
the protocol accepts them without modification and its conclusions
are not an artifact of self-implemented probes."""

new = """To check that the benchmark is not self-referential, we ran
AERCA~\\cite{han2025aerca}, a published Granger-causal attribution
method, end-to-end with its official implementation and training
budget (5{,}000 epochs with the built-in early stopping,
MSDS-style configuration, GPU) on six RCAEval fault cases, the full
SWaT injected unit, and three SCM trajectories (870 windows in
total), scoring its encoder-latent deviations under both axes of
our protocol. \\emph{Validity}: on RCAEval, per-case hit@3 is
0.60/0.26/0.03/0.00/0.00/0.00 (mean 0.15), far below the simplest
deviation reader on the same testbed (0.78); on SCM it is
competitive with the graph families (0.51--0.69); on SWaT it lands
mid-pack (0.53). \\emph{Stability}: under input-noise perturbation
of the scored windows (5--10\\% of channel standard deviation), its
top-3 ACR@3 stays between 0.83 and 0.995 on every RCAEval and SCM
unit, so most of its low-validity configurations are simultaneously
highly stable: the published method itself occupies the stable-wrong
quadrant. The exception is instructive: on SWaT its ACR@3 collapses
to 0.23, i.e.\\ a method's stability is as testbed-dependent as its
validity, which is this paper's point made by an external method.
(A shorter 400-epoch pilot had scored one RCAEval case at 1.00; the
official-budget run reverses it, a reminder that headline numbers
for learned methods are budget-sensitive.) Broad grid-wide
evaluation of published methods remains future work; the point here
is that the protocol accepts them without modification and its
conclusions are not an artifact of self-implemented probes."""
assert t.count(old) == 1, 'AERCA block not found'
t = t.replace(old, new, 1)

old2 = "AERCA (official implementation,\nepoch-capped) enters on the two real injected testbeds"
new2 = ("AERCA (official implementation and\nbudget) enters on all four testbeds with both metrics")
assert t.count(old2) == 1, 'limitations anchor'
t = t.replace(old2, new2, 1)
open(r'D:\0科研\工作1\第11篇SCI\paper\DKE_StabilityNotValidity.tex', 'w',
     encoding='utf-8').write(t)

cl_path = r'D:\0科研\工作1\第11篇SCI\04_投稿准备\CoverLetter_DKE.txt'
cl = open(cl_path, encoding='utf-8').read()
old3 = ("benchmark also runs AERCA (ICLR 2025), a published end-to-end\n"
        "root-cause method, under the same protocol on both real injected\n"
        "testbeds, so the evaluation is not limited to self-implemented probes.")
new3 = ("benchmark also runs AERCA (ICLR 2025), a published end-to-end\n"
        "root-cause method, with its official training budget on all four\n"
        "testbeds and both metrics (validity and label-free stability); the\n"
        "published method itself occupies the stable-wrong quadrant on most\n"
        "RCAEval cases, so the evaluation is neither self-referential nor\n"
        "favorable only to our probes.")
assert old3 in cl, 'CL anchor'
open(cl_path, 'w', encoding='utf-8').write(cl.replace(old3, new3, 1))
print('e50 integrated + limitations + CL')
