# -*- coding: utf-8 -*-
"""Figure: ARP protocol flow diagram (双指标审计流程图)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

DARK = '#3F345B'
BLUE = '#7FA4C8'
RED = '#D08AB6'
GREY = '#8E8E8E'
LIGHT = '#F0EDF5'

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 4.5)
ax.axis('off')

def box(x, y, w, h, text, color=BLUE, fontsize=9, text_color='white'):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                       facecolor=color, edgecolor='none', zorder=2)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color=text_color, fontweight='bold', zorder=3)

def arrow(x1, y1, x2, y2, color=GREY):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

# Row 1: Inputs
box(0.2, 3.5, 2.3, 0.7, 'Uncertain Graph\n$G_0$ (learned)', BLUE, 8)
box(2.8, 3.5, 2.0, 0.7, 'Scorer $f$\n(iforest/PCA/AE)', BLUE, 8)
box(5.1, 3.5, 2.3, 0.7, 'Anomalous Window\n$w$ (flagged by $f$)', BLUE, 8)
box(7.6, 3.5, 2.0, 0.7, 'Attribution\nMethod $\\mathcal{A}$', BLUE, 8)

# Row 2: Perturbation
box(0.7, 2.3, 3.5, 0.7, 'Perturbation Protocol\n(del/add/rew + noise/slice)', GREY, 8)
box(4.5, 2.3, 2.0, 0.7, '$M$ Perturbed\nInstances', GREY, 8)

# Row 3: Dual metric
box(1.0, 1.0, 2.5, 0.7, 'Stability: ACR@$K$\n(label-free)', RED, 9)
box(3.8, 1.0, 2.5, 0.7, 'Validity: hit@$K$\n(ground truth)', RED, 9)
box(6.6, 1.0, 2.2, 0.7, 'Envelope\nCertificate', BLUE, 8)
box(9.0, 1.0, 2.5, 0.7, 'Reliability\nPredictor', BLUE, 8)

# Row 4: Output
box(3.0, 0.0, 6.0, 0.6, 'ARP: Pre-Deployment Check  →  surface / withhold / human review', DARK, 9)

# Arrows
arrow(1.35, 3.4, 1.35, 3.1)  # graph -> perturbation
arrow(3.8, 3.4, 3.8, 3.1)   # scorer -> perturbation
arrow(2.45, 2.2, 2.25, 1.8)  # perturbation -> ACR
arrow(5.5, 2.2, 5.0, 1.8)   # instances -> hit
arrow(3.6, 1.35, 3.7, 1.35)  # ACR -> hit (comparison)
arrow(6.4, 1.35, 6.5, 1.35)  # hit -> certificate
arrow(8.9, 1.35, 8.9, 1.35)  # certificate -> predictor
arrow(5.0, 0.9, 5.5, 0.7)   # metrics -> ARP
arrow(7.7, 0.9, 7.0, 0.7)   # instruments -> ARP
arrow(8.6, 3.4, 8.6, 1.8)   # method -> predictor (features)

# Decoupling label
ax.text(3.6, 1.85, '⇍ decouple ⇏', ha='center', fontsize=7, color=DARK,
        style='italic')

fig.tight_layout()
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'paper', 'figures', 'fig_protocol.pdf')
fig.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
print('-> figures/fig_protocol.pdf')
