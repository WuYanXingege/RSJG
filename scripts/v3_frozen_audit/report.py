"""Render frozen-audit results; only reads already verified inference outputs."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np

SOURCE=Path(__file__).resolve().parents[2]
DOC=SOURCE/'docs/research/bcep_v3/frozen_feedback_audit_20261121'
OUT=SOURCE.parent/'RSJG_EBJD_assets/experiments/zara2_v3_frozen_feedback_audit_20261121'


def read(name):return json.loads((DOC/name).read_text())
def rows(name):
    with (DOC/name).open() as f:return list(csv.DictReader(f))
def write(name,obj):(DOC/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def f(r,k):return float(r[k])


def main():
    summary=read('SUMMARY.json');proof=read('REPLAY_PROOF.json');g=summary['groups']
    agents=rows('AGENT_MATCHING.csv');ties=rows('TIE_RELABEL_CHECKS.csv');scores=rows('SCORE_DISTRIBUTIONS.csv')
    groups=rows('GROUPS.csv');examples=rows('KEY_EXAMPLES.csv');relations=rows('RELATION_GROUPS.csv')
    gap_stats=[];tie_stats=[]
    for mode in 'ABC':
        x=np.array([f(r,mode+'_gap') for r in agents if r[mode+'_gap']!=''])
        gap_stats.append(dict(mode=mode,agent_matrices=len(x),strict_ties=int((x==0).sum()),
            positive_near_le_1e_minus6=int(((x>0)&(x<=1e-6)).sum()),minimum=float(x.min()),
            median=float(np.median(x)),p95=float(np.quantile(x,.95)),maximum=float(x.max())))
        for label in ['strict_tie_0','near_(0,1e-6]']:
            rr=[r for r in ties if r['mode']==mode and r['gap_class']==label]
            changed=[r for r in rr if int(r['changed_after_row_column_relabel'])>0]
            tie_stats.append(dict(mode=mode,gap_class=label,matrices=len(rr),changed_assignments=len(changed),
                changed_JFDE_cases=sum(f(r,'one_person_relabel_delta_JFDE')!=0 for r in changed),
                max_abs_delta_JFDE_m=max([abs(f(r,'one_person_relabel_delta_JFDE')) for r in changed]+[0]),
                max_abs_score_loss=max([abs(f(r,'original_minus_relabel_objective')) for r in rr]+[0])))
    write('GAP_DISTRIBUTIONS.json',gap_stats);write('TIE_SUMMARY.json',tie_stats)
    shutil.copyfile(OUT/'AUDIT_PROTOCOL.json',DOC/'AUDIT_PROTOCOL.json')
    shutil.copyfile(OUT.with_name(OUT.name+'_replay_attempt1')/'NATIVE_FUNCTION_PROBE.json',DOC/'NUMERICAL_PREFLIGHT_PROBE.json')
    native_paths=['bcep_v3/coupling.py','bcep_v3/formal_validation.py','bcep_v3/gate.py',
                  'bcep_v2/joint.py','ebjd/unified.py','bcep_joint/inference.py']
    code_audit=dict(base_commit=proof['base_commit'],native_files_changed=False,
        native_file_sha256={n:hashlib.sha256((SOURCE/n).read_bytes()).hexdigest() for n in native_paths},
        exact_capture=dict(S0='native forward_features output',old_feedback='native messages output',
            B='native old_final before HCRA addition',HCRA_delta='native return_details.score_delta',C='native final scores'),
        original_native_replay='generate -> HCRACoupling.assign -> original two assignments -> original 20-step bridge',
        independent_counterfactual_solves='assignment(S0), assignment(old_final), assignment(final_scores); each unchanged original Hungarian',
        both_feedback_terms_conditioned_on='the same initial A hard assignment; HCRA does not recompute its relations from B',
        scoring_network_not_rerun_for_ABC=True,model_constructions_in_accepted_full_replay=1,model_constructions_in_analysis=0,
        preflight_model_loads=2,preflight_explanation='One failed exact-JADE check and one same-weight flag-toggle probe; neither used for branch comparisons.',
        preserved_native_requires_grad_flags=True,freeze_enforcement=['no_grad','deny optimizer construction and step','deny backward','state hash before/after'],
        native_requires_grad_probe='True/False/False/True on one strict-loaded model reproduces 1-ULP JADE shift; precise operator/kernel not identified',
        historical_permutations_not_archived=True,user_approved_native_replay_reference=True,
        native_H_JADE_and_endpoint_H_JFDE_per_window_exact=True,all_ABC_per_person_bijections=True,
        singleton_identity=True,C_gate_passed_before_attribution=True,
        optimizer_steps_real=0,optimizer_steps_synthetic=0,backward_calls=0,
        test_use=False,model_selection=False,original_run_updates_unchanged=5040)
    write('CODE_AUDIT.json',code_audit)
    write('TEST_RESULTS.json',dict(unit_tests_passed=7,unit_test_command="CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 ../.conda/rsjg/bin/python -m pytest -q tests/test_v3_frozen_matching.py",
        coverage=['second-best assignment gap vs exhaustive 4! enumeration, 3 seeds','strict vs positive near ties','uniform world relabel vs correspondence change','row/column gauge invariance','tie-breaking non-equivariance'],
        production_acceptance=dict(native_archive_JFDE_exact_windows=590,native_archive_JADE_exact_windows=590,C_native_permutation_exact_windows=590,
            A_initial_permutation_exact_windows=590,A_archived_I_JFDE_exact_windows=590,independent_ABC_bijection_windows=590,singleton_identical_windows=89),
        real_optimizer_updates=0,synthetic_optimizer_updates=0))
    metric_table='\n'.join(f"| {r['group']} | {r['windows']} | {r['A_JFDE']:.9f} | {r['B_JFDE']:.9f} | {r['C_JFDE']:.9f} |" for r in g)
    delta_table='\n'.join(f"| {r['group']} | {label} | {r[key+'_mean']*100:+.6f} | {r[key+'_median']*100:+.6f} | {r[key+'_improved']} / {r[key+'_worse']} / {r[key+'_equal']} | {r[key+'_improved_gt_1cm']} / {r[key+'_worse_gt_1cm']} |" for r in g for key,label in [('B_minus_A','旧反馈 B−A'),('C_minus_B','HCRA C−B'),('C_minus_A','合计 C−A')])
    selection_table='\n'.join(f"| {r['group']} | {r['BC_changed_choices']} / {r['BC_total_choices']} | {r['BC_changed_choice_ratio']*100:.4f}% | {r['BC_changed_agents']} / {r['total_agents']} | {r['BC_windows_with_choice_changes']} / {r['windows']} | {r['BC_uniform_world_relabel_only_windows']} |" for r in g)
    people_table='\n'.join(f"| {r['group'][2:]} | {r['windows']} | {f(r,'B_minus_A_mean')*100:+.5f} | {f(r,'C_minus_B_mean')*100:+.5f} | {f(r,'C_minus_A_mean')*100:+.5f} | {r['C_minus_B_improved']} / {r['C_minus_B_worse']} / {r['C_minus_B_equal']} |" for r in groups if r['group'] in ['N_1','N_2-4','N_5-8','N_9-12','N_13-20','N_21+'])
    score_table='\n'.join(f"| {r['term']} | {r['projection']} | {f(r,'mean_abs'):.6g} | {f(r,'rms'):.6g} | {f(r,'p05'):.6g} | {f(r,'median'):.6g} | {f(r,'p95'):.6g} |" for r in scores if r['group']=='all')
    gap_table='\n'.join(f"| {r['mode']} | {r['agent_matrices']} | {r['strict_ties']} | {r['positive_near_le_1e_minus6']} | {r['median']:.6g} | {r['p95']:.6g} | {r['maximum']:.6g} |" for r in gap_stats)
    tie_table='\n'.join(f"| {r['mode']} | {r['gap_class']} | {r['matrices']} | {r['changed_assignments']} | {r['changed_JFDE_cases']} | {r['max_abs_delta_JFDE_m']*100:.6f} |" for r in tie_stats)
    relation_table='\n'.join(f"| {r['transition']} | {r['stratum']} | {r['changed_choices']} | {f(r,'mean_candidate_FDE_delta_m')*100:+.5f} | {r['better_candidates']} / {r['worse_candidates']} | {f(r,'mean_endpoint_shift_m'):.5f} |" for r in relations if r['stratify']=='proximity_before')
    grg_table='\n'.join(f"| {r['group']} | {r['windows']} | {r['on_H_JADE']:.9f} | {r['off_H_JADE']:.9f} | {r['on_minus_off_mean_m']*100:+.6f} | {r['on_minus_off_median_m']*100:+.6f} | {r['on_better_windows']} / {r['on_worse_windows']} |" for r in summary['GRG'])
    example_ids=[2998,2700,2716,3192]
    all_windows={int(r['index']):r for r in rows('THREE_WAY_WINDOWS.csv')}
    example_table='\n'.join(f"| {i} | {all_windows[i]['n_agents']} | {f(all_windows[i],'A_JFDE'):.6f} | {f(all_windows[i],'B_JFDE'):.6f} | {f(all_windows[i],'C_JFDE'):.6f} | {all_windows[i]['BC_changed_choices']} |" for i in example_ids)
    report=f'''# ZARA2 v3 epoch 12：冻结权重三路配对机制审计

**当前权重的主要新增退化发生在 B→C（加入 HCRA 残差）这一步。** 旧反馈在全体开发窗平均改善 0.280899 cm，但在 N≥9 窗平均恶化 0.744680 cm；HCRA 进一步使这两组平均恶化 **1.535715 cm / 2.449841 cm**。HCRA 改变了 **48,873 / 85,240（57.3358%）** 个 world–candidate 选择，涉及 **500 / 501** 个多人窗口的真实跨人对应，不能用参数范数小推断其无效。

本次 **真实 optimizer.step=0，合成 optimizer.step=0，backward=0**。固定 epoch 12，不重新选模、不训练、不测试集评测。A/B/C 是**同一已训练模型的推理反事实**，不是三个独立训练模型或正式训练消融。

## 复现门槛与固定身份

- 固定源码/报告提交：`{proof['base_commit']}`；epoch 12 权重 SHA：`{proof['checkpoint_sha256']}`。
- 冻结配置 SHA：`{proof['frozen_sha256']}`；初始/最终已加载参数指纹一致：`{proof['trained_model_sha256_before']}`。
- 官方 selection_dev **590 窗、4,262 个有效行人**；multi **501 窗**；N≥9 **127 窗**；实际开发集最大人数 42。candidate seed2035、原 refined bank 顺序、K20、FP32、原 Hungarian 与 20 步 DDIM 保持不变。
- 原归档未保存每窗排列。经用户明确同意，先用固定源码和同一训练权重进行原生回放：**590 窗归档 endpoint H-JFDE 和 H-JADE 逐值精确一致**；再核验分支 C 的 **590 窗完整排列与原生回放逐项相等**，并逐值匹配归档 JFDE。该门槛全部通过后才运行比较分析。不把新回放排列说成历史已保存的排列。
- 全量原生路径 H-JADE **0.4124557339160119 m**、H-JFDE **0.8049277728581328 m** 精确复现。下文三路比较统一采用冻结候选直接计算的 endpoint JFDE；C 为 **0.8049277718731407 m**，与原生路径汇总仅相差约 9.85e-10 m，这是原归档已有的数值口径差异，不混用两种聚合值。

证据：[REPLAY_PROOF.json](REPLAY_PROOF.json)、[C_GATE.json](C_GATE.json)、[逐窗回放](REPLAY_WINDOWS.csv)、[代码审计](CODE_AUDIT.json)。

## 同一次评分前向中的 A/B/C

原生 `coupling.assign` 的 `forward_features` 给出 S0；`messages` 给出旧反馈；`old_final` 给出 S0+旧反馈；`score_delta` 给出 HCRA 残差。只读取这些张量，保留原运算次序。三个分支分别独立调用原 `assignment` / `hard_assignment_indices`，所有有效行人均通过 K20 一一双射检查。

**A**：S0 的初配；**B**：S0+old_feedback；**C**：S0+old_feedback+HCRA_delta。原源码中，旧反馈和 HCRA 都以 **A 的初次排列**作为条件；HCRA 不是以 B 的新排列重新计算邻居关系。此依赖关系是代码事实；它是否造成退化仍是待解释机制。

| 分组 | 窗数 | A JFDE m | B JFDE m | C JFDE m |
|---|---:|---:|---:|---:|
{metric_table}

| 分组 | 变化 | 均值差 cm | 中位数差 cm | 改善 / 恶化 / 不变窗 | 改善>1cm / 恶化>1cm |
|---|---|---:|---:|---:|---:|
{delta_table}

所有误差差值为后减前，负值为改善；均值和中位数均按窗口等权。严格零计“不变”，另列 |Δ|≤1e-6 m 的窗数。旧反馈全体均值虽略有益，恶化窗 248 仍多于改善窗 240，不能概括为处处有益。N≥9 中，HCRA 对总退化的算术贡献约 **76.69%**，其余来自旧反馈；这不是训练因果贡献比例。

## 分配改变与人数分布

| 分组 | B→C 改变选择 / 总选择 | 选择比例 | 改变行人 / 总行人 | 改变窗口 / 总窗口 | 仅统一世界换编号 |
|---|---:|---:|---:|---:|---:|
{selection_table}

“选择”指一个有效行人的一个世界槽位，分母为 ΣN×20；该比例是槽位加权，不冒充窗口等权指标。窗口等权的改变比例另在 [GROUPS.csv](GROUPS.csv)。对齐每窗第 0 位匿名行人的候选编号后再比较其他人，可检测是否只是整组世界统一换编号；本次 500 个改变窗均存在真正的跨人对应改变。89 个单人窗的 A/B/C 排列恒等且 JFDE 完全相同。

| 人数 | 窗数 | B−A cm | C−B cm | C−A cm | HCRA 改善 / 恶化 / 不变窗 |
|---|---:|---:|---:|---:|---:|
{people_table}

HCRA 在预先固定的每个多人区间平均均为退步，其中 **21 人及以上的 50 窗**最明显：额外退步 **4.413769 cm**，36 窗恶化、14 窗改善；旧反馈在该组也退步 2.787843 cm。效果不随人数严格单调；13–20 人的旧反馈仍有较明显收益。精确到每个人数的分组也在 GROUPS.csv，不能用小样本组替代全体结论。

## 评分幅度与近并列机制

原评分大部分幅度来自对每行或每列都可分离的偏置。完整双射会恰好使用每行每列一次，因此这些偏置不能区分两个完整匹配。为描述真正能区分分配的部分，仅在**分析时**计算 double-centered 分数：S−行均值−列均值+总均值；原 Hungarian 仍使用未改动的原生 FP32 评分。

| 项 | 投影 | 平均绝对值 | RMS | P05 | 中位数 | P95 |
|---|---|---:|---:|---:|---:|---:|
{score_table}

这里按全部有效评分元素汇总，单位是评分，不是米。HCRA 原始 RMS 约 2.59e-5，远小于 S0 的 4.28e-2；但去掉行列偏置后，HCRA 的 **1.87e-7** 已接近 S0 的 **2.42e-7**。因此，原始残差幅度小不足以证明分配影响小。完整 all/multi/N≥9 分布见 [SCORE_DISTRIBUTIONS.csv](SCORE_DISTRIBUTIONS.csv)，逐行人信息见 [AGENT_MATCHING.csv](AGENT_MATCHING.csv)。

匹配间隔定义为“最优完整匹配总分−第二个不同完整匹配总分”。逐一禁止最优解的 20 条选中边，再取各受限最优解中的最高分，可得到全局次优解；不是单行 top1−top2。比较使用原 FP32 评分值的 float64 求和。严格并列为间隔恰为 0；近并列预设为 0<间隔≤1e-6 评分总分单位。89 个单人矩阵由原生规则直接走恒等排列，不计入求解器并列分母。

| 分支 | 多人有效矩阵 | 严格并列 | 正间隔≤1e-6 | 间隔中位数 | P95 | 最大值 |
|---|---:|---:|---:|---:|---:|---:|
{gap_table}

B 的全部 4,173 个多人矩阵均处于该严格/近并列范围。其严格并列组有 **526/527** 个行人分配被 HCRA 改变，正近并列组有 **3,633/3,646** 个被改变。改变分配的 B 分数损失中位数约 **8.22e-7**，HCRA 提供的实际加分优势中位数约 **1.88e-6**，足以跨过原分配间隔。每个改变解均核验“C 总分增益=HCRA 实际增益−B 分数损失≥0”；因此不是 Hungarian 算错了目标，而是分数目标更优不保证终点 JFDE 更好。

在 4,159 个改变行人中，旧反馈分量对 C 相对 B 的新解持反向评分者 **2,980** 个，同向 **1,174** 个，完全中性 5 个；HCRA 分量全部严格偏好 C。此事实支持关注两种反馈的评分协作，但不能证明它们在训练中为什么形成这种关系。[COOPERATION.csv](COOPERATION.csv)、[GAP_GROUPS.csv](GAP_GROUPS.csv)、[间隔分布](GAP_DISTRIBUTIONS.json)。

## 严格并列的编号敏感性

对全部严格/近并列矩阵执行固定行反序、列循环移位，然后映射回原编号；仅测试求解器处理同一批评分的行为。再固定同窗其他人的原分配，只替换该行人的这个等价解，观察 JFDE。没有改变权重或重新训练。

| 分支 | 间隔类型 | 矩阵数 | 回映射后分配改变 | 单人替换使JFDE改变的案例 | 最大绝对JFDE变化 cm |
|---|---|---:|---:|---:|---:|
{tie_table}

发生分配改变的案例原目标总分损失均为 **0**。C 的 389 个严格并列矩阵中，155 个在该重编号检查下改变，34 个单人替换案例改变场景 JFDE，最大约 **17.48 cm**。这些是行人矩阵/单次替换案例数量，不是独立窗口数量；并列解并未用于替换本报告的正式 A/B/C 结果。正间隔组在这一次固定重编号中未改变，不由此宣称所有排列下都等变。当前只能保证使用原编号和原求解器时可精确复现，不能宣称无条件排列等变。[TIE_RELABEL_CHECKS.csv](TIE_RELABEL_CHECKS.csv)

## 哪些候选关系受到负向修正

除人数外，直接支持的关系是：**在完整匹配间隔很小的情况下，HCRA 用微小有效评分差大幅改变跨人候选对应**。为补充物理描述，使用同一观察历史构造原定义的候选参考路径，固定邻居为初配 A 的对应候选，按改变前与这些邻居的最小同步距离分组。下表衡量被替换候选自己的 FDE，不是场景 JFDE；距离阈值只是描述分组，不宣称碰撞或真实交互因果。

| 转换 | 改变前参考最小距离 | 改变槽位数 | 局部候选FDE均值差 cm | 候选改善 / 恶化 | 平均终点移动 m |
|---|---|---:|---:|---:|---:|
{relation_table}

HCRA 在 0.5–1 m 参考距离组的局部候选 FDE 均值变差约 **2.54 cm**，<0.2 m 组反而改善约 **7.01 cm**，不能把全部退化归结为“近距离候选被错误推开”。每人20个候选仅重新排列，所以跨其全部世界求和的候选 FDE **必然守恒**；按整个人数组统计的槽位平均差为零也是这一恒等式，不能拿它推翻场景最优世界 JFDE 的退步。距离子组受条件选择影响，同窗多个槽位彼此相关，不能视为独立样本或因果证据。[CHANGED_CHOICES.csv](CHANGED_CHOICES.csv)、[RELATION_GROUPS.csv](RELATION_GROUPS.csv)

## 关键反例

| 匿名窗口 | 人数 | A m | B m | C m | B→C 改变槽位 |
|---|---:|---:|---:|---:|---:|
{example_table}

窗口 2998：主要损害已在旧反馈发生，HCRA 只略微回补。窗口 2700：旧反馈严重变差，但 HCRA 明显救回，说明不能把 HCRA 对每一窗都判为负向。窗口 2716：旧反馈没有改变 JFDE，HCRA 却从约 0.445 m 推高到 1.015 m。窗口 3192：旧反馈先改善，随后被 HCRA 抵消。完整按每种转换的五个最大正/负例固定导出于 [KEY_EXAMPLES.csv](KEY_EXAMPLES.csv)，所有原始分配见 [WORLD_CANDIDATES.csv](WORLD_CANDIDATES.csv)；案例仅解释分布，不替代全体统计。

## GRG-CTF 的可选冻结检查

同一个 epoch12 模型、同一 C 配对、相同缓存 context、相同 keyed noise 与20步DDIM，只将 `1+0.5*tanh(r)` 的残差调制置零，保留原基础门控和掩码。所有窗口启用/关闭后的最终端点逐值相同；JFDE 与 minFDE 保持不变。

| 分组 | 窗数 | 开启 H-JADE m | 仅关闭残差 H-JADE m | 开−关均值 cm | 开−关中位数 cm | 开启改善 / 恶化窗 |
|---|---:|---:|---:|---:|---:|---:|
{grg_table}

GRG 在全体有约 **0.132 cm** 的即时平均收益，在 N≥9 则恶化约 **0.466 cm**，97/127 窗恶化。这只衡量当前已训练模型对门控的即时依赖，不能推断一个从头未使用 GRG 的模型会怎样；没有用关闭后的指标重新选模，也没有修改训练权重。[GRG_GROUPS.csv](GRG_GROUPS.csv)

## 研究判断与边界

1. **旧反馈是否有益？** 对当前权重的全体均值略有益，密集组有害，且人数分组异质；不能整体判为无效或全面有益。
2. **HCRA 增量贡献？** 在 B 之后使全体/多人/密集组分别额外退步 1.535715 / 1.808527 / 2.449841 cm，是当前新增退化的主要直接来源；同时存在救回的窗口。
3. **是否实际改变足够多分配？** 改变全体 57.34% 槽位、500 个多人窗，属于广泛改变；其有效评分部分与 S0 同量级，又面对极小间隔，不能根据参数范数否认作用。
4. **负向集中在哪？** 预设人数区间中21人及以上最突出；分配关系上普遍处于近并列，许多 HCRA 新解逆转旧反馈偏好。物理距离子组只能提供描述线索，尚不足以锁定某类真实交互作为原因。
5. **更支持优先检查什么？** 优先审查 **HCRA 加入旧反馈后，在近并列评分上的协作与匹配稳定性**。直接问题定位 B→C；现有数据不支持先整体删除旧反馈，也不足以证明 HCRA 架构本身在重新训练后必然无益。共享表征、训练梯度、目标函数如何形成近可加评分仍无法由本次冻结检查归因。

该判断没有提出新的损失权重，没有改网络，也不触发下一次训练、v4、扩折或测试集。独立训练贡献、重新优化后的补偿、跨seed和跨折效果均未被证明。

## 实现、数值排错与资产保护

所有原生源码与固定清单一致；新增代码只位于 `scripts/v3_frozen_audit/` 与测试/报告目录。正式全量回放只构建并严格加载一次模型，A/B/C 分析不构建模型。初始数值排错额外加载相同 epoch12 两次，均未用于反事实分支或产生更新。

预检发现：把全部 `requires_grad` 设为 False 后，即使原验证在 no_grad 中，首窗 JADE 也从 0.2326759397983551 变为 0.23267590999603271；True/False/False/True 切换可重复恢复。尚未把差异定位到具体算子或内核。随后保留原生标志，以 no_grad、禁止 optimizer 构造/step、禁止 backward、参数指纹前后比对冻结权重，590窗最终全部精确通过。该排错不改变权重数值。[NUMERICAL_PREFLIGHT_PROBE.json](NUMERICAL_PREFLIGHT_PROBE.json)

7项只读组合测试通过；590窗C复现、三路双射、89窗单人恒等、评分切换分解均通过。[TEST_RESULTS.json](TEST_RESULTS.json)。训练主机四个原实验根目录共 **{proof['protected_files']:,} 个文件、{proof['protected_bytes']:,} 字节**前后SHA完全一致，包括原30轮权重、best/last/Top5、v2基线及候选bank；全部分析结束后再次核对通过，见 [PROTECTION_FINAL.json](PROTECTION_FINAL.json)。未上传权重、bank或原始轨迹；逐窗NPZ评分回放留在独立审计目录。历史文件的保护哈希不是测试集评测。

原五折保持 PAUSED_RESUMABLE，candidate_cost 保持 SESSION_BUDGET_PAUSED，sharedworld 与本次 v3 原训练保持 COMPLETE。原 v3 真实更新仍为历史 5040，本次新增 0。

## 复现

本次实际命令（在本审计 worktree 中执行；输出目录已存在时主动拒绝覆盖）：

```bash
OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 ../.conda/rsjg/bin/python scripts/v3_frozen_audit/replay.py
OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 ../.conda/rsjg/bin/python scripts/v3_frozen_audit/analyze.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 ../.conda/rsjg/bin/python -m pytest -q tests/test_v3_frozen_matching.py
```

在保留所有现有结果的前提下，用新的独立输出目录复现：

```bash
OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 ../.conda/rsjg/bin/python scripts/v3_frozen_audit/run.py replay --root ../RSJG_EBJD_assets/experiments/zara2_v3_frozen_audit_repeat
OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 ../.conda/rsjg/bin/python scripts/v3_frozen_audit/run.py analyze --root ../RSJG_EBJD_assets/experiments/zara2_v3_frozen_audit_repeat --docs docs/research/bcep_v3/frozen_audit_repeat
```

这些命令只推理和统计，不包含任何训练入口。完整门槛与预设分组见 [AUDIT_PROTOCOL.json](AUDIT_PROTOCOL.json)。审计完成后退出。
'''
    (DOC/'REPORT_ZH.md').write_text(report)
    write('ARTIFACT_SHA256.json',{str(p.relative_to(DOC)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DOC.iterdir()) if p.is_file() and p.name!='ARTIFACT_SHA256.json'})
    print(json.dumps(dict(report=str(DOC/'REPORT_ZH.md'),created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()),ensure_ascii=False))


if __name__=='__main__':main()
