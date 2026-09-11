# -*- coding: utf-8 -*-
"""一键复现入口: 顺序运行全部实验脚本 + 表/图生成.
用法:
  python scripts/reproduce_all.py             # 全部(CPU; 跳过GPU项)
  python scripts/reproduce_all.py --with-gpu  # 含 AERCA 官方预算(很慢)
  python scripts/reproduce_all.py --from e39  # 从某实验续跑
  python scripts/reproduce_all.py --only e57_innovation,e58_rcaeval_acr
输出: _cache/reproduce_log.json + 控制台进度; 单脚本失败不中断, 记录后继续.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import sys
import json
import time
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
CACHE = os.path.join(SRC, '_cache')

# (脚本名, 说明, 是否需要GPU)
ALL = [
    ('e1_anchor', '锚点实验: SCM 主网格 + 象限分解'),
    ('e2_full', '全量网格 v1'),
    ('e3_predictor', '窗口级预测器 + 特征表'),
    ('e4_mitigation', '缓解: 集成/置信加权 v1'),
    ('e6_round2', '轮2: 关系轴'),
    ('e8_round6', '轮6: 网格扩展'),
    ('e10_swaT_testbed', 'SWaT 测试床'),
    ('e11_boost', '增强/边界结果'),
    ('e12_predictor_boost', '预测器增强'),
    ('e13_gnn_ae', 'GNN 探针'),
    ('e14_swat_ci_pred', 'SWaT CI 预测'),
    ('e15_round15', '轮15 综合'),
    ('e17_gnn_positive', 'GNN 正例'),
    ('e18_graa', 'GRAA v1'),
    ('e18b_graa_v2', 'GRAA v2'),
    ('e19_smd', 'SMD 补充'),
    ('e20_graa_swat', 'GRAA SWaT'),
    ('e21_graa_v3', 'GRAA v3'),
    ('e22_graa_v4', 'GRAA v4'),
    ('e23_graa_swat_fast', 'GRAA SWaT 快速'),
    ('e24_graa_te', 'GRAA TE + 源交换'),
    ('e25_graa_signif', 'GRAA 显著性'),
    ('e26_recalib', '重校准'),
    ('e26b_targetonly', '目标域-only'),
    ('e27_knee_fine', '膝点细扫'),
    ('e27b_merge', '合并分析'),
    ('e28_graa_sens', 'GRAA 敏感性'),
    ('e29_graa_rel', 'GRAA 关系轴'),
    ('e30_graa_scorers', 'GRAA×打分器'),
    ('e31_rcaeval', 'RCAEval 第四床(hit)'),
    ('e32_graa_clean', 'GRAA clean'),
    ('e33_graa_full', 'GRAA 全量+种子'),
    ('e34_mconv', 'M收敛'),
    ('e35_wlci', '窗口级CI'),
    ('e35b_ci', 'CI 补充'),
    ('e36_ae_real', '真实数据AE'),
    ('e37_edges', '边验证'),
    ('e37b_model', '模型对照'),
    ('e38_selector', '选择器'),
    ('e39_cert_coverage', '证书覆盖曲线'),
    ('e40_te_dense', 'TE 密集变体'),
    ('e41_runtime', '运行时'),
    ('e42_calib_holm', '校准+Holm'),
    ('e43_case', '案例研究'),
    ('e44_aerca', 'AERCA 外部基线', True),
    ('e45_gate_agg', '门控聚合'),
    ('e46_effect', '效应分解'),
    ('e47_envelope', '包络违反率'),
    ('e48_topo_scale', '拓扑缩放'),
    ('e49_seeds10', '十种子'),
    ('e50_aerca_full', 'AERCA 全量(官方预算)', True),
    ('e51_snr', 'SNR 扫描'),
    ('e52_purged_cv', '净化时间CV'),
    ('e53_adaptive_p', '自适应p'),
    ('e54_cases2', '案例2'),
    ('e55_aerca_native', 'AERCA 原生协议'),
    ('e56_dissect', 'GRAA 解剖'),
    ('e56b_gatefix', '门控修正'),
    ('e57_innovation', '创新增量 B/A/D'),
    ('e57b_topo_sw', '拓扑拟合+SW网格'),
    ('e57c_swrate_cells', 'SW率阈值敏感性'),
    ('e58_rcaeval_acr', 'RCAEval ACR 闭环'),
    ('e59_adversarial', '对抗边选择'),
    ('e60_prop2', 'Prop2 裕度校准'),
    ('e61_disagree', '跨方法分歧(负)'),
    ('e61b_disagree_clean', '分歧复测(学图)'),
    ('e62_te_cert', 'TE 经验证书'),
    ('e64_review_addendum', '外部评审补充计算'),
    ('gen_tables', '生成全部表格'),
    ('make_sci_figures', '生成全部图'),
    ('make_fig_certcov', '证书覆盖图'),
]


def main():
    args = sys.argv[1:]
    with_gpu = '--with-gpu' in args
    start = None
    only = None
    if '--from' in args:
        start = args[args.index('--from') + 1]
    if '--only' in args:
        only = set(args[args.index('--only') + 1].split(','))

    log = []
    skip_note = ('GPU 项跳过(加 --with-gpu 运行)' if not with_gpu else '')
    started = start is None
    for item in ALL:
        name = item[0]
        gpu = len(item) > 2 and item[2]
        if only is not None and name not in only:
            continue
        if start is not None and name == start:
            started = True
        if not started:
            continue
        if gpu and not with_gpu:
            print(f'[SKIP-GPU] {name} {skip_note}', flush=True)
            log.append(dict(name=name, status='skip-gpu', sec=0))
            continue
        t0 = time.time()
        print(f'[RUN] {name} ...', flush=True)
        r = subprocess.run([sys.executable, '-u',
                            os.path.join(HERE, name + '.py')],
                           cwd=SRC, capture_output=True, text=True,
                           timeout=7200)
        sec = round(time.time() - t0, 1)
        ok = r.returncode == 0
        print(f'[{"OK" if ok else "FAIL"}] {name} {sec}s', flush=True)
        if not ok:
            print(r.stderr[-1500:], flush=True)
        log.append(dict(name=name, status='ok' if ok else 'fail', sec=sec))
        json.dump(log, open(os.path.join(CACHE, 'reproduce_log.json'), 'w'),
                  indent=1)
    n_ok = sum(1 for x in log if x['status'] == 'ok')
    n_fail = [x['name'] for x in log if x['status'] == 'fail']
    print(f'\n=== 完成: {n_ok} 成功, {len(n_fail)} 失败 ===')
    if n_fail:
        print('失败项:', n_fail)


if __name__ == '__main__':
    main()
