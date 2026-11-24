"""Report a single fixed dev experiment; never load trajectories or tune a rule."""
from bcep.contracts import digest
from bcep_joint.common import write


def csvwrite(path,rows):
    import csv
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def retention_text(check):
    if check['retention_percent'] is None:
        return 'N/A（A≤B，无B正收益；要求R≤A）'
    return f"{check['retention_percent']:.9f}%"


def build_report(results,config,destination):
    destination.mkdir(parents=True,exist_ok=True)
    all_summary=[];all_checks=[];all_windows=[]
    both_marginal=True
    for value in results.values():
        d=value['diagnostics']
        both_marginal &= (d['inverse_restore_failures']==0 and d['agent_minADE_max_abs_m']==0
                          and d['agent_minFDE_max_abs_m']==0 and d['singleton_metric_failure_windows']==0)
    passed=bool(both_marginal and all(v['decision']['passed'] for v in results.values()))
    conclusion='值得后续独立验证' if passed else '未达到预定判据，结束本次重排方案'
    failures=[]
    overview=['# 用 B 场景编号重排 A 完整轨迹：两折开发集配对实验',
        f'**最终结论：{conclusion}。** 结论限定为 **ZARA1/ZARA2 两折开发集、单推理种子2035**。',
        '逐人边缘 minADE、minFDE '+('精确保留' if both_marginal else '未通过恒等核验')+'；此结论来自同窗A/R原生逐人指标逐项比较，不是由汇总近似判断。']
    if config.get('recovery'):
        overview.append('回放修复说明：最初CPU补推理的A在ZARA1归档一致性检查中停止；24个观测窗的设备核查中，11个多人窗CPU/GPU的H0排列不同、13个单人窗一致。恢复时固定使用原确定性CUDA配置重放A，仍复用原B；未改变权重、种子、匹配代价或判据。旧CPU预测和已生成的671窗ZARA1评分留在训练主机，未用于最终判定。修复后的两折全部预测重新封存并核验后才读取真值评分；本次不是新的规则或参数试验。')
    for fold,value in results.items():
        metrics=value['metrics'];diag=value['diagnostics'];decision=value['decision']
        all_summary.extend(value['summary']);all_checks.extend(decision['joint_checks'])
        rates='；'.join(f"{x['group']} {x['metric']}：{retention_text(x)}" for x in decision['joint_checks'])
        overview.append(f"{fold.upper()} 的 B 联合收益保留率：{rates}。本折判据：{'通过' if decision['passed'] else '未通过'}。")
        lines=[f'# {fold.upper()}：A / B / B指导重排R',
            '两折开发集、单推理种子2035。训练更新0，测试集访问0；关闭整组交换、关闭1 cm校准；只运行预测终点欧氏距离这一种匹配代价。',
            f"边缘恒等性：逆置换还原失败 {diag['inverse_restore_failures']} 次；逐人 minADE 最大绝对差 {diag['agent_minADE_max_abs_m']:.12g} m；逐人 minFDE 最大绝对差 {diag['agent_minFDE_max_abs_m']:.12g} m。单人窗 {diag['singleton_windows']} 个，四项指标不一致 {diag['singleton_metric_failure_windows']} 个。",rates,
            f"本折固定判据：{'通过' if decision['passed'] else '未通过'}。",'## 全体和密集组原生指标',
            '绝对值为米；差值统一为R减比较模型，单位厘米。联合指标按窗等权，边缘指标按有效行人人次加权。',
            '| 分组 | 窗数 | 人次 | 指标 | A（m） | B（m） | R（m） | R−A（cm） | R−B（cm） |',
            '|---|---:|---:|---|---:|---:|---:|---:|---:|']
        for row in value['summary']:
            lines.append(f"| {row['group']} | {row['windows']} | {row['agents']} | {row['metric']} | {row['A']:.12f} | {row['B']:.12f} | {row['R']:.12f} | {row['R_minus_A_cm']:+.9f} | {row['R_minus_B_cm']:+.9f} |")
        lines+=['## 未舍入判据',
            '若A>B，保留率=(A−R)/(A−B)，通过界为R≤B+0.10×(A−B)。保留负值及超过100%的比例。若A≤B，比例为N/A，要求R≤A。下表R−上限为正即未通过。',
            '| 分组／指标 | 保留率 | R（m） | 判据上限（m） | R−上限（cm） | 判定 |',
            '|---|---:|---:|---:|---:|---|']
        for check in decision['joint_checks']:
            lines.append(f"| {check['group']} / {check['metric']} | {retention_text(check)} | {check['R']:.12f} | {check['limit_m']:.12f} | {check['R_minus_limit_cm']:+.9f} | {'通过' if check['passed'] else '未通过'} |")
            if not check['passed']:
                failures.append(dict(fold=fold,group=check['group'],metric=check['metric'],R_minus_limit_m=check['R_minus_limit_m'],R_minus_limit_cm=check['R_minus_limit_cm'],retention_percent=check['retention_percent']))
        lines+=['## 缓存和实际耗时',
            f"复用原B封存完整轨迹 {diag['B_reused_prior_windows']} 窗，B补推理 {diag['B_new_inference_windows']} 窗；原A已有可匹配身份的完整轨迹缓存 {diag['A_reused_prior_full_trajectory_windows']} 窗，补齐原生H0完整轨迹 {diag['A_missing_windows_inferred']} 窗。A只补推理缺项，重跑可复用本次已落盘的原生A缓存。R不再解码，额外解码 {diag['R_decode_calls']} 次。",
            f"匹配总耗时 {diag['matching_seconds']:.6f} s（含终点代价计算、匈牙利、完整轨迹搬移及逆置换核验）；其中linear_sum_assignment求解调用累计 {diag['hungarian_seconds']:.6f} s。A原生补解码累计 {diag['A_native_decode_seconds']:.3f} s。实际轨迹数组改变 {diag['changed_windows']} 窗、零变化 {diag['zero_change_windows']} 窗，均纳入计分。",
            f"最终A运行设备为{diag['A_device']}；修复前CPU原生A轨迹保留{diag['prior_cpu_native_A_windows_retained_for_audit']}窗用于审计，最终A与旧CPU H0排列不同的窗口为{diag['cpu_cuda_H0_changed_windows']}个。若存在设备修复，旧CPU推理不计入最终预测补齐耗时，修复原因和既有真值读取记录见总报告及RUN_CONFIG.json。",
            'A采用原生mode=H配对并解码，K20、DDIM20、eta=0。B来自固定上轮缓存NPZ中的原B数组，取末帧作为匹配目标，未使用swap数组。两路用完整split/member哈希、冻结模型哈希、seed和每窗有效人数对齐。B的所有场景编号保持原顺序；各人匈牙利矩阵行是B编号、列是A轨迹编号，匹配后全体行人共享同一个B场景编号轴。',
            f"匈牙利求解器：SciPy {config['scipy_version']}，float64欧氏距离，原编号顺序，不加随机扰动。输出R每个坐标直接取自A；未再次修正终点或运行扩散桥。",
            '本次最终版本的两折全部A/B/R预测封存且文件逐项SHA核验后，评分端才读取开发集未来，用原生hard_metric_statistics和MetricAccumulator计分。逐窗统计缓存支持恢复而不改变规则。B原生聚合与固定缓存归档精确一致；A原生CUDA回放与历史归档的差异另见A_BASELINE_REPLAY.json，仍使用预先固定的1e−4 m核对容差。ZARA1密集组A为本轮实际轨迹评分，不用其他组代替。',
            '[汇总CSV](SUMMARY.csv) · [匿名逐窗CSV](PER_WINDOW.csv) · [六项判据中的本折三项](JOINT_CRITERIA.csv) · [恒等性与耗时](INVARIANTS_AND_RUNTIME.json) · [固定资产](ASSET_IDENTITY.json) · [评分收据](SCORE_RECEIPT.json)']
        (destination/fold/'README.md').write_text('\n\n'.join(lines).replace('|\n\n|','|\n|')+'\n')
        import csv
        with (destination/fold/'PER_WINDOW.csv').open(newline='') as f:all_windows.extend(csv.DictReader(f))
    overview+=['## 边缘恒等性',
        '| 折 | 逆置换失败 | 逐人minADE最大绝对差（m） | 逐人minFDE最大绝对差（m） | 单人窗指标失败 |',
        '|---|---:|---:|---:|---:|']
    for fold,v in results.items():
        d=v['diagnostics'];overview.append(f"| {fold.upper()} | {d['inverse_restore_failures']} | {d['agent_minADE_max_abs_m']:.12g} | {d['agent_minFDE_max_abs_m']:.12g} | {d['singleton_metric_failure_windows']} |")
    if failures:
        overview+=['## 未达标项','以下均使用未舍入值判断。正差值表示R超过预定距离上限。','| 折 | 分组 | 指标 | R−判据上限（cm） |','|---|---|---|---:|']
        for f in failures:overview.append(f"| {f['fold'].upper()} | {f['group']} | {f['metric']} | {f['R_minus_limit_cm']:+.9f} |")
    overview+=['## 交付',
        '[ZARA1中文报告](zara1/README.md) · [ZARA2中文报告](zara2/README.md) · [两折汇总CSV](SUMMARY.csv) · [全部匿名逐窗CSV](PER_WINDOW.csv) · [固定判据CSV](JOINT_CRITERIA.csv) · [最终判定](DECISION.json)',
        '只评估这一种匹配规则，不追加种子、代价或参数搜索。权重、原始轨迹、置换矩阵和行人ID保留在训练主机；公开CSV不含原始坐标或行人ID。四折控制器继续原计划，本实验未更改其训练、选模或测试规则。']
    (destination/'README.md').write_text('\n\n'.join(overview).replace('|\n\n|','|\n|')+'\n')
    csvwrite(destination/'SUMMARY.csv',all_summary);csvwrite(destination/'JOINT_CRITERIA.csv',all_checks);csvwrite(destination/'PER_WINDOW.csv',all_windows)
    write(destination/'RUN_CONFIG.json',config)
    write(destination/'DECISION.json',dict(scope='two-fold selection_dev, single inference seed2035',marginal_identity=bool(both_marginal),all_six_joint_checks_pass=passed,conclusion=conclusion,failed_items=failures,joint_checks=all_checks,training_updates=0,test_access=0))
    write(destination/'ARTIFACT_SHA256.json',{str(p.relative_to(destination)):digest(p) for p in sorted(destination.rglob('*')) if p.is_file() and p.name!='ARTIFACT_SHA256.json'})
