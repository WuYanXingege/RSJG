"""Archive the predeclared endpoint gate, including a negative result."""
import csv
import shutil
from joint_flow_fourfold.report import public_metadata
from .common import RUN,DOC,BASE,read,write,digest

LABEL={'all':'全体','all_low_speed':'全员低速','mixed':'动静混合','all_moving':'全员运动',
       'mixed_N2_8':'2–8人混合','all_moving_N2_8':'2–8人全员运动'}


def build_report():
    DOC.mkdir(parents=True,exist_ok=True)
    config=read(RUN/'RUN_CONFIG.json');receipt=read(RUN/'SCORE_RECEIPT.json')
    decision=read(RUN/'DECISION.json');bounds=read(RUN/'BOUND_CHECK.json');replay=read(RUN/'NATIVE_REPLAY.json')
    metrics=read(RUN/'METRICS.json');seal=read(RUN/'PREDICTIONS_FROZEN.json')
    assert receipt['status']=='COMPLETE' and replay['passed'] and bounds['passed']
    # A passing endpoint gate is not the experiment's final decision.
    assert not decision['endpoint_gate_passed'],'endpoint gate passed: run authorized full DDIM continuation before final archive'
    table=list(csv.DictReader((RUN/'SUMMARY.csv').open()));wall=read(RUN/'WALL_TIME.json')
    allgroup=metrics['all'];imp=100*decision['improvement_m'];worst=100*bounds['max_actual_deterioration_m']
    lines=['# HOTEL：带逐窗 JFDE 保护的整场单次替换',
        f"**三个结果：全体真实 minFDE 改善 {imp:.6f} cm；最差单窗 JFDE 恶化 {worst:.6f} cm；688 窗中实际替换 {decision['replaced_windows']} 窗。**",
        f"**停在第1步，归档并结束本次路线。** 全体 minFDE 改善未达到预定2 cm门槛，距离门槛还差{100*(.020-decision['improvement_m']):.6f} cm。没有放宽0.005 m保护预算，没有扫参，未进入DDIM或ZARA1/ZARA2。当前数据不支持按本轮规则继续。",
        '本轮只用 HOTEL selection_dev 全部688窗、5203人次。冻结B第19轮、seed=2035、M=256、K=20；B原有桥前CandidateHead完整保留。训练更新0、测试访问0、DDIM调用0。四折A/B主线与既有权重保持原状态。',
        '**差值统一为新方案−原B；负值表示误差下降。** 个人覆盖目标P只用于候选排序，真实minFDE/JFDE仅在全部预测封存后评分。',
        '## 固定算法与保护范围',
        '每窗从原生20个ID出发，枚举20×236个整场单次替换，所有行人共用换入ID并占原槽。用CPU float64计算P=mean_{m,i} min_{k∈S} ||X[m,i]−X[k,i]||，仅保留严格正收益，按收益降序、换出原始ID及换入原始ID升序排序。一次最多接受一个替换；未通过者原样返回B。',
        '严格按排序逐个考虑所有正收益候选，直到首个通过者；不使用近似头输出筛掉候选。每个候选集合均重算冻结CandidateHead的组内均值与全部20组输出。CUDA Graph只摊薄重复调用开销，保留原生张量计算体与矩阵形状，不改代数表达或批量GEMM形状。',
        '最终接受方案另用原生头输出复核。方向距离h=max_a min_b mean_i ||Y0[a,i]−Y[b,i]||以CPU FP64计算，仅接受h≤0.005 m；接受预算不加任何容差。',
        '由三角不等式，对任意终点真值，JFDE_new−JFDE_B≤h。这是JFDE单向恶化上限；不承诺JFDE绝对变化、个人minFDE或整条轨迹JADE的界。',
        '## 真实终点评分',
        'minFDE按有效人次加权，JFDE按窗口等权；每项独立从同一套20组输出取最优。绝对值单位米，差值单位厘米。',
        '| 分组 | 窗数/人次 | 指标 | B(m) | 新方案(m) | 新−B(cm) |',
        '|---|---:|---|---:|---:|---:|']
    for row in table:
        values=f"{float(row['B_m']):.12f} | {float(row['new_m']):.12f} | {float(row['delta_cm']):+.6f}" if row['B_m'] else 'N/A | N/A | N/A'
        lines.append(f"| {LABEL[row['group']]} | {row['windows']}/{row['agents']} | {row['metric']} | {values} |")
    lines+=['观测速度沿用前8帧相邻7段路程之和÷2.8秒；v≤0.2 m/s为低速。动静混合同时包含低速与高于阈值者，全员运动表示全部行人高于阈值。分组只用于分析，没有参与选择或门槛调整。',
        '## 替换与逐窗保护核验',
        '| 分组 | 实际替换窗 | 无正收益窗 | 有正收益但无可行替换窗 |','|---|---:|---:|---:|']
    for group,value in metrics.items():
        lines.append(f"| {LABEL[group]} | {value['replaced_windows']} | {value['no_positive_gain_windows']} | {value['no_feasible_windows']} |")
    lines += [f"最大h={bounds['max_h_m']:.12f} m；实测最大ΔJFDE={bounds['max_delta_JFDE_m']:.12f} m。",
        '| 核验 | 不加容差超界窗数 | 加FP32绝对容差1e−5 m后超界窗数 |','|---|---:|---:|',
        f"| ΔJFDE≤h | {bounds['delta_exceeds_h_strict_windows']} | {bounds['delta_exceeds_h_tolerance_windows']} |",
        f"| ΔJFDE≤0.005 m | {bounds['delta_exceeds_budget_strict_windows']} | {bounds['delta_exceeds_budget_tolerance_windows']} |",
        f"保护接受门槛本身没有加容差：h>0.005 m的封存窗口数为{bounds['h_exceeds_budget_windows']}。FP64重新计分的max(ΔJFDE−h)={bounds['max_delta_fp64_minus_h_m']:.12g} m；原生FP32计分的对应最大值为{bounds['max_delta_minus_h_m']:.12g} m。",
        '## 原B回放与封存顺序',
        '| 指标 | 历史B(m) | 本轮B(m) | 差(m) |','|---|---:|---:|---:|']
    for row in replay['checks']:lines.append(f"| {row['metric']} | {row['reference_m']:.12f} | {row['actual_m']:.12f} | {row['difference_m']:+.12g} |")
    lines += ['回放核验使用既定atol=rtol=1e−6，全部通过。每窗旧端点、原生ID、观测和有效掩码均与上一轮封存缓存逐元素一致。',
        f"全体预测封存时间={receipt['global_seal_unix']}；完整校验完成={receipt['verification_unix']}；首次读取开发集真值={receipt['first_truth_read_unix']}。三者严格按此顺序，评分共读取688窗开发真值。候选提议、排序、接收和回退都不读取未来。",
        '## 身份、运行与归档',
        f"起始提交`{BASE}`。实际B依赖权重为HOTEL第19轮，SHA256=`{config['identity']['teacher_sha256']}`；上游和冻结状态哈希见RUN_CONFIG.json。",
        '上一轮真实原始256组坐标、原生20个ID、修正后端点和观测复用688窗；原始候选生成0次。未用20个ID或覆盖摘要重建原始256组。',
        f"本次运行及评分墙钟{wall['wall_seconds']:.3f}秒（{wall['wall_seconds']/60:.2f}分钟），CPU评分{receipt['score_wall_seconds']:.3f}秒。固定配置、实际命令、封存清单和评分收据随报告归档。",
        '原始候选、完整轨迹、真实人员ID和权重保留在训练主机。公开逐窗表仅含匿名窗号、人数、分组、选择诊断量和误差。',
        '[分组CSV](SUMMARY.csv) · [匿名逐窗CSV](PER_WINDOW.csv) · [固定配置](RUN_CONFIG.json) · [停止判定](DECISION.json) · [保护界核验](BOUND_CHECK.json) · [原B回放](NATIVE_REPLAY.json) · [封存清单](PREDICTIONS_FROZEN.json) · [评分收据](SCORE_RECEIPT.json)']
    (DOC/'README.md').write_text('\n\n'.join(lines).replace('|\n\n|','|\n|')+'\n')
    for name in ('SUMMARY.csv','PER_WINDOW.csv'):shutil.copyfile(RUN/name,DOC/name)
    for name in ('RUN_CONFIG.json','SCORE_RECEIPT.json','PRE_SCORE_VERIFICATION.json','NATIVE_REPLAY.json','BOUND_CHECK.json',
                 'METRICS.json','DECISION.json','WALL_TIME.json','COMMANDS.json','FROZEN_MODELS.json','CACHE_AUDIT.json','CHECKS.json','NATIVE_GRAPH_PARITY.json'):
        if (RUN/name).exists():write(DOC/name,public_metadata(read(RUN/name)))
    public_seal=dict(seal,rows=[{k:v for k,v in row.items() if k!='index'} for row in seal['rows']],
                     host_full_manifest_sha256=digest(RUN/'PREDICTIONS_FROZEN.json'))
    write(DOC/'PREDICTIONS_FROZEN.json',public_metadata(public_seal))
    write(DOC/'ARTIFACT_SHA256.json',{str(p.relative_to(DOC)):digest(p) for p in sorted(DOC.rglob('*')) if p.is_file() and p.name!='ARTIFACT_SHA256.json'})
    return decision
