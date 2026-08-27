# -*- coding: utf-8 -*-
"""一键复现入口: 按依赖顺序运行全部实验脚本并重建表/图.

用法:  python reproduce_all.py            # 全量 (CPU 数小时)
       python reproduce_all.py --core     # 仅主表+关键实验 (~1h)
线程自动限制在 4 (OMP/MKL), 不占满机器.
"""
import os
import subprocess
import sys

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

HERE = os.path.dirname(os.path.abspath(__file__))

CORE = [
    'e1_anchor', 'e11_boost', 'e15_round15', 'e22_graa_v4',
    'e33_graa_full', 'e35_wlci', 'e35b_ci', 'e25_graa_signif',
    'e39_cert_coverage', 'e40_te_dense', 'e41_runtime',
    'e42_calib_holm', 'e43_case', 'e45_gate_agg', 'e46_effect',
    'e47_envelope', 'e49_seeds10',
]
EXTRA = [
    'e2_full', 'e3_predictor', 'e6_round2', 'e8_round6', 'e9_round7',
    'e10_swaT_testbed', 'e19_smd', 'e24_graa_te', 'e26_recalib',
    'e26b_targetonly', 'e27_knee_fine', 'e28_graa_sens', 'e29_graa_rel',
    'e30_graa_scorers', 'e31_rcaeval', 'e32_graa_clean', 'e34_mconv',
    'e36_ae_real', 'e37_edges', 'e37b_model', 'e38_selector',
    'e44_aerca', 'e48_topo_scale', 'e51_snr',
]
POST = ['gen_tables', 'gen_boost_tables', 'make_sci_figures']


def run(name):
    path = os.path.join(HERE, 'scripts', name + '.py')
    if not os.path.exists(path):
        print(f'[skip] {name}')
        return
    print(f'=== {name} ===', flush=True)
    subprocess.run([sys.executable, path], check=False)


if __name__ == '__main__':
    todo = CORE if '--core' in sys.argv else CORE + EXTRA + POST
    for name in todo:
        run(name)
    print('\nDone. Tables/figures are in paper/tables and paper/figures.')
