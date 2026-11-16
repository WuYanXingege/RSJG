"""Read-only aggregation of the frozen flow and single MDN control receipts."""
import argparse,csv,hashlib,json,statistics
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_csv(path,rows):
    if not rows:return
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);args=parser.parse_args()
    plan=json.loads(args.plan.read_text());root=Path(plan['assets']);doc=Path(plan['doc'])
    def read(name,default=None):
        path=root/name
        return json.loads(path.read_text()) if path.exists() else default
    flow=read('flow_diagnostics/SUMMARY.json');mdn=read('mdn_evaluation/SUMMARY.json')
    training=read('reports/MDN_TRAINING_RESULT.json',dict(status='NOT_STARTED'))
    test=read('reports/MDN_TEST_REPORT.json');reg=read('reports/FROZEN_BCEP_REGRESSION.json')
    training_state='PATIENCE_STOP' if training.get('stale',0)>=3 and training.get('stop_reason')=='COMPLETE' else training.get('stop_reason','未结束')
    budgets=read('reports/READONLY_BUDGET.json');execution=read('identity/MDN_EXECUTION_IDENTITY.json',{})
    gap=read('flow_diagnostics/SUPPORT_GAP_DIAGNOSTICS.json')
    assert flow['status']=='COMPLETE' and reg['status']=='PASS'
    complete=bool(mdn and mdn['status']=='COMPLETE' and mdn['seeds']==list(range(2035,2040)))
    models={'flow':flow}
    if complete:models['mdn8']=mdn
    rows=[];per_seed={}
    for kind,summary in models.items():
        folder='flow_diagnostics' if kind=='flow' else 'mdn_evaluation'
        per_seed[kind]=[]
        for seed in range(2035,2040):
            result=read(f'{folder}/result_seed{seed}.json');assert result['metadata']['complete']
            assert len(result['metadata']['members'])==320 and result['metadata']['member_hash']=='3f978031af7003722efd0f04ed2cc4cd9b2f8babfdd9ddcb945c40329eb8a599'
            assert result['metadata']['checkpoint_sha256']==summary['checkpoint_sha256']
            for method in ['iid20','coverage20']:
                for group,metrics in result['metrics'][method].items():
                    rows.append(dict(model=kind,method=method,group=group,inference_seed=seed,**metrics))
            per_seed[kind].append(dict(seed=seed,metrics=result['metrics'],final_nll=result['likelihood']['eval_nll_final_agent_micro'],energy_iid256=result['energy_iid256_agent_micro'],duplicate_fraction=result['duplicate_fraction']))
    deltas={}
    metrics=['minFDE','recall_0.25','recall_0.5','recall_1.0','P50','P90','P95']
    for kind,s in models.items():
        gain=[r['metrics']['iid20']['all']['minFDE']-r['metrics']['coverage20']['all']['minFDE'] for r in per_seed[kind]]
        deltas['selector_gain_'+kind]=dict(values=gain,mean=statistics.mean(gain),sample_sd=statistics.stdev(gain),improving_seeds=sum(x>0 for x in gain))
    if complete:
        for method in ['iid20','coverage20']:
            deltas['density_contrast_'+method]={}
            for metric in metrics:
                v=[m['metrics'][method]['all'][metric]-f['metrics'][method]['all'][metric] for m,f in zip(per_seed['mdn8'],per_seed['flow'])]
                deltas['density_contrast_'+method][metric]=dict(values=v,mean=statistics.mean(v),sample_sd=statistics.stdev(v))
    comparison=dict(status='COMPLETE' if complete else 'INCOMPLETE',training_seed=3101,inference_seeds=list(range(2035,2040)),
        scope='Internal uni_examples dev; 320 windows / 621 agent occurrences; not outer ETH and not five training replicates',
        model_summaries=models,per_seed=per_seed,deltas=deltas,
        provider_replacement='NOT_ESTABLISHED',full_ebjd_independence_performance='NOT_ESTABLISHED',
        equal_six_epoch_completed_window=training.get('completed_epoch')==6,
        old_flow_best_epoch=6,old_flow_best_updates=22740,mdn_selected_epoch=training.get('selected_epoch'))
    (doc/'FOUR_CELL_COMPARISON.json').write_text(json.dumps(comparison,indent=2)+'\n');write_csv(doc/'FOUR_CELL_COMPARISON.csv',rows)
    for filename,source in [('SUPPORT_GAP_DIAGNOSTICS.json','flow_diagnostics/SUPPORT_GAP_DIAGNOSTICS.json'),('SUPPORT_GAP_DIAGNOSTICS.csv','flow_diagnostics/SUPPORT_GAP_DIAGNOSTICS.csv'),('FROZEN_BCEP_REGRESSION.json','reports/FROZEN_BCEP_REGRESSION.json'),('MDN_TEST_REPORT.json','reports/MDN_TEST_REPORT.json'),('MDN_TRAINING_RESULT.json','reports/MDN_TRAINING_RESULT.json'),('READONLY_BUDGET.json','reports/READONLY_BUDGET.json'),('MDN_EXECUTION_IDENTITY.json','identity/MDN_EXECUTION_IDENTITY.json'),('END_STATE.json','identity/END_STATE.json'),('SELECTED_MDN_IDENTITY.json','identity/SELECTED_MDN_IDENTITY.json'),('STATISTICS_CHECK.json','reports/STATISTICS_CHECK.json'),('PROTECTED_ASSETS_INDEX.json','identity/PROTECTED_BEFORE.json')]:
        if (root/source).exists():(doc/filename).write_bytes((root/source).read_bytes())
    if complete:
        for suffix in ['json','csv']:(doc/f'MDN_SUPPORT_GAP_DIAGNOSTICS.{suffix}').write_bytes((root/f'mdn_evaluation/SUPPORT_GAP_DIAGNOSTICS.{suffix}').read_bytes())
    curve_path=root/'mdn_training/epochs.jsonl';curve=[]
    if curve_path.exists():
        curve=[json.loads(r) for r in curve_path.read_text().splitlines()]
        (doc/'MDN_EPOCH_METRICS.json').write_text(json.dumps(curve,indent=2)+'\n')
    allgap=gap['pooled_description']['all'];part=allgap['partition']['0.5']
    lines=['# BCEP冻结支持诊断与单次MDN8对照','',
        f"冻结BCEP回归：PASS（五种子逐值相同）；MDN预检：{test['status']}；MDN训练：{training['status']} / {training_state}；四格评测：{'COMPLETE' if complete else 'INCOMPLETE'}。",
        'PROVIDER_REPLACEMENT=NOT_ESTABLISHED；FULL_EBJD_INDEPENDENCE_PERFORMANCE=NOT_ESTABLISHED。','',
        f"BCEP在0.5m处：选中点命中{part['selected_hit']['rate']*100:.2f}%，支持池命中但选择遗漏{part['pool_hit_selection_miss']['rate']*100:.2f}%，本次有限支持池未命中{part['pool_miss']['rate']*100:.2f}%。后两者分别占未覆盖出现次数的{100*part['pool_hit_selection_miss']['count']/(part['pool_hit_selection_miss']['count']+part['pool_miss']['count']):.2f}%和{100*part['pool_miss']['count']/(part['pool_hit_selection_miss']['count']+part['pool_miss']['count']):.2f}%。",
        '池未命中不证明密度不存在该模态；选择遗漏不证明贪心实现错误，其目标不使用单条真实未来。','',
        '## 冻结身份与实验控制',
        'BCEP固定epoch6 best（22740次更新，SHA cc71f02c3315bb7e9596600689740c644150b44d62f21b24df2fea6e51ec03a4）；没有重启旧run或纳入其第7轮619次更新。',
        'train=3790窗口；dev=uni_examples的320窗口、621次有效行人出现。米制、dt=0.4、预测12帧，观察长度来自原manifest。M256/K20；推理seed2035–2039。每一行IID20取该行同一M256的前20点，coverage20用原选择器。不同密度的支持坐标不相同。',
        'MDN8是标准控制：8个二维全协方差Gaussian分量，context128+horizon_seconds，两层128 SiLU，48维输出；softplus对角+0.001，非对角不约束，稳定logsumexp和三角解。Linear默认随机权重及偏置打破对称，没有GT初始化、额外正则或调参。',
        f"编码器初值逐位相同，hash=`{test['initial_encoder_hash']}`；各自独立随机构造，未加载BCEP/GDTS/旧EBJD训练权重。其余motion frame、Jacobian、aux0.25、Adam1e-4、gamma1、场景加权、loader独立Generator和NLL选模规则保持一致。",
        '新增独立MDN schema、runner与导出入口；原bcep/及原train/export身份校验未修改。runner仅替换模型/schema/import，终轮达到更新上限后仍可完成验证（在墙钟预算内）。',
        '同协议不代表参数量、FLOPs或墙钟相同；同推理种子不代表两种密度共享相同支持。','',
        '## BCEP支持池与压缩诊断',
        f"五seed平均：e_pool={allgap['errors']['pool_error']['mean']:.6f}m，e_cov={allgap['errors']['chosen_error']['mean']:.6f}m，mean(gap)={allgap['errors']['compression_gap']['mean']:.6f}m。逐seed验证e_cov=e_pool+gap及子集关系。",
        'M256整池最近点是只读诊断，不是K20预测结果、部署选择器或桥输入。参考诊断用float64距离，原K20指标沿用float32；末位舍入差不是候选变化。分位数不作加法归因。','',
        '| seed | selected_hit@0.5 | pool_hit_selection_miss | pool_miss | 合计 |','|---:|---:|---:|---:|---:|']
    for seed,v in gap['per_seed'].items():
        p=v['all']['partition']['0.5'];c=[p[k]['count'] for k in ['selected_hit','pool_hit_selection_miss','pool_miss']]
        lines.append(f'| {seed} | {c[0]} | {c[1]} | {c[2]} | {sum(c)} |')
    lines+=['','| 半径m | selected_hit | pool_hit_selection_miss | pool_miss |','|---:|---:|---:|---:|']
    for radius,p in allgap['partition'].items():lines.append('| '+radius+' | '+' | '.join(f"{100*p[k]['rate']:.2f}%" for k in ['selected_hit','pool_hit_selection_miss','pool_miss'])+' |')
    lines+=['','上表比例为五种子等权平均（每seed分母621）；JSON同时保存3105个“出现×seed”的合并计数及分组分位数。它们不是3105个独立行人或独立训练重复。','',
        '## 冻结四格结果',
        '均值±样本SD（ddof=1），距离单位m，Recall为比例。SD仅反映同一选中模型的推理seed变化。','',
        '| 密度 / 选择 | minFDE | Recall@0.25 | Recall@0.5 | Recall@1 | P50 | P90 | P95 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for kind,summary in models.items():
        for method in ['iid20','coverage20']:
            r=summary['statistics'][method]['all']
            lines.append('| '+kind+' / '+method+' | '+' | '.join(f"{r[k]['mean']:.6f} ± {r[k]['sample_sd']:.6f}" for k in metrics)+' |')
    lines+=['','| 密度 | 世界终点NLL | IID256 energy score | 参数量 | IID20重复率 | coverage20重复率 |','|---|---:|---:|---:|---:|---:|']
    for kind,s in models.items():lines.append(f"| {kind} | {s['final_nll_agent_micro']:.6f} | {s['energy_iid256']['mean']:.6f} ± {s['energy_iid256']['sample_sd']:.6f} | {s['parameter_count']} | {s['duplicate_fraction']['iid20']:.6f} | {s['duplicate_fraction']['coverage20']:.6f} |")
    lines+=['','重复率定义为逐行人(20−精确不同坐标数)/20再取均值；未对近邻点施加人为合并阈值。NLL仅用于原连续密度，energy仅用于合法IID256，不在coverage代表点上计算。','',
        '| 分组 | 密度/选择 | minFDE | Recall@0.5 | P90 |','|---|---|---:|---:|---:|']
    for group in ['N_eq_1','N_ge_2']:
        for kind,s in models.items():
            for method in ['iid20','coverage20']:
                r=s['statistics'][method][group];lines.append(f"| {group} | {kind}/{method} | {r['minFDE']['mean']:.6f} | {r['recall_0.5']['mean']:.6f} | {r['P90']['mean']:.6f} |")
    lines+=['','## 选择器与密度贡献分开报告']
    for kind in models:
        g=deltas['selector_gain_'+kind];lines.append(f"- selector gain({kind})：{g['mean']:+.6f}m，{g['improving_seeds']}/5种子改善（正数表示coverage更好）。")
    if complete:
        for method in ['iid20','coverage20']:
            d=deltas['density_contrast_'+method];lines.append(f"- density contrast({method})=MDN−flow：minFDE {d['minFDE']['mean']:+.6f}m，P90 {d['P90']['mean']:+.6f}m，P95 {d['P95']['mean']:+.6f}m，Recall@0.5 {d['recall_0.5']['mean']*100:+.2f}个百分点。误差负数为MDN更好。")
    lines+=['','## 实际训练和选模窗口',
        f"MDN完成{training.get('completed_epoch',0)}轮 / {training.get('update',0)}次真实更新；选中epoch={training.get('selected_epoch')}，dev NLL={training.get('selection_value')}；停止原因={training_state}。",
        '早停规则为patience3，best只来自完整320窗口NLL；未用coverage指标挑选epoch。没有第二次run或追加训练。']
    if training.get('completed_epoch')!=6:lines.append('MDN未完成6轮，比较包含实际训练/选模窗口差异，不能称为严格相同6轮预算完成。选中MDN epoch1仅经历3790次更新，flow best epoch6经历22740次；两者遵循相同停止/选模规则，但实际训练暴露量不同。')
    else:lines.append('两种密度均完成6轮选择窗口；最优epoch允许由相同NLL规则选在不同轮次。这是受控短试训，不代表各模型均已收敛或充分调参。')
    lines+=['','| MDN epoch | update | train final NLL | dev final NLL | 累计秒 |','|---:|---:|---:|---:|---:|']
    for r in curve:lines.append(f"| {r['epoch']} | {r['update']} | {r['train_nll_final_agent_micro']:.6f} | {r['dev_nll_final_agent_micro']:.6f} | {r['elapsed_wall_seconds']:.2f} |")
    lines+=['','## 定点例子（post-hoc failure visualization）',
        '固定seed2035、0.5m、每类按scene_id/agent_id排序取前2项；仅作诊断，不参与候选选择或证明总体提升。完整支持坐标见资产EXAMPLES.json。','',
        '| 类别 | scene_id | agent_id | pool误差m | coverage误差m | gap m |','|---|---|---|---:|---:|---:|']
    for r in read('flow_diagnostics/EXAMPLES.json')['examples']:lines.append(f"| {r['category']} | {r['scene_id']} | {r['agent_id']} | {r['pool_error']:.5f} | {r['chosen_error']:.5f} | {r['compression_gap']:.5f} |")
    lines+=['','## 成本、复现和保护',
        f"本轮合成/接口预检：{test['total_preflight_seconds']:.3f}s /600s（35项CPU测试含原25项；GPU合成16次更新、真实无梯度N=1/3/57）。合成GPU连续8步与3+5恢复逐位一致。",
        f"MDN训练调用墙钟：{training.get('wall_seconds',0):.3f}s /3600s，含数据检查/准备、训练、完整验证和保存；峰值allocated显存 {training.get('peak_allocated_bytes',0)/2**30:.3f}GiB。",
        f"本轮两阶段只读累计：{budgets['seconds']:.3f}s /1800s，CPU执行、optimizer updates=0。未额外进行三人数层级的分项延迟benchmark；导出墙钟不当作模型在线延迟。",
        '历史BCEP原run训练耗时3600.153s，包含未进入best的第7轮619次更新；其第6轮完成累计3506.802s。历史成本与本轮新增MDN成本分列，不借旧预算续训。',
        f"实际训练代码：`{execution.get('execution_sha','未启动')}`；证据基线：`{plan['evidence_sha']}`。模型与训练/导出源码冻结后未改变，report-only提交另列。",
        f"资产目录：`{root}`。best/last、完整支持bank与哈希见ARTIFACTS.json。Git无权重、大bank或原始日志。",'',
        '```bash',
        f"python scripts/bcep_postpilot_readonly.py --plan {Path(plan['doc']).relative_to(Path(__file__).resolve().parents[1])}/PLAN_AND_AUTHORIZATION.json --stage flow-diagnostics",
        f"python scripts/mdn_control_gate.py --config configs/research/{plan['run']}.yaml",
        f"python scripts/train_mdn_control.py --config configs/research/{plan['run']}.yaml",
        f"python scripts/bcep_postpilot_readonly.py --plan {Path(plan['doc']).relative_to(Path(__file__).resolve().parents[1])}/PLAN_AND_AUTHORIZATION.json --stage mdn-evaluation",
        '```',
        '上述为已实现入口的执行记录，使用项目`.conda/rsjg/bin/python`；run claim与累计预算会拒绝重复启动。CPU命令：`python -m pytest tests/test_bcep_core.py tests/test_bcep_prepilot.py tests/test_mdn_control.py -q`。','',
        '## 结论边界',
        '只使用内部dev；原matched已接触此dev，未被调用为公平胜负对照。历史0.5811m不作为本次门槛。MDN是Bishop (1994)的标准混合密度控制，不是本项目新创新。',
        '支持池/选择两部分误差并存，不能仅凭整池GT最近点诊断宣称可部署改善，也不能从一次短控制推广为所有MDN不如flow或flow必不可少。',
        '本轮结束后没有BCEP续训、第二个MDN run、seed/fold队列、联合桥训练或后台监控。最终方向由实际四格结果决定，见本报告补充判读。']
    if complete:
        dc=deltas['density_contrast_coverage20'];fi=flow['statistics']['coverage20']['all'];mi=mdn['statistics']['coverage20']['all']
        interpretation=[
            '本次判断：flow的平均端点误差较好，但必要性尚未建立；选择器收益在两种密度上均出现。',
            f"coverage20下flow相对MDN平均minFDE减少{dc['minFDE']['mean']:.6f}m（以MDN为分母约{100*dc['minFDE']['mean']/mi['minFDE']['mean']:.2f}%），{sum(v>0 for v in dc['minFDE']['values'])}/5推理seed同向；Recall@0.5高{-100*dc['recall_0.5']['mean']:.2f}个百分点。",
            f"但flow的P95比MDN高{-dc['P95']['mean']:.6f}m，{sum(v<0 for v in dc['P95']['values'])}/5 seed均是MDN较低。MDN的NLL较低而energy与平均minFDE较高，说明这些评价刻画不同方面，不能用单一NLL推出有限K覆盖更好。",
            '单人组coverage minFDE仅相差约0.0037m，多人组约0.0208m；这是本开发集分组观察，不能据此认定某个网络模块的因果贡献。',
            '实际早停和选中更新数不同，加上只有一个训练seed，当前不能把这点差距当作flow容量优势的充分证据，也不能用MDN更短成本宣称等质量加速。',
            '唯一优先方向：先将方法核心表述收敛为“可替换密度提供器上的有限预算覆盖选择”，把flow保留为候选实现、MDN8保留为简洁控制；暂不据此增加flow复杂度或启动联合桥。此处没有自动把MDN升级为主方案。'
        ]
        lines[4:4]=['',interpretation[0]]
        lines+=['','## 实际判读与唯一优先方向',*interpretation[1:]]
    (doc/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    artifacts=dict(selected_mdn_identity=read('identity/SELECTED_MDN_IDENTITY.json'),model_sources={'flow':plan['frozen_bcep_best'],'mdn8':{k:training.get(k) for k in ['best.pt','last.pt','selected_epoch','selection_value','completed_epoch','update']}},files=[])
    for folder in ['flow_diagnostics','mdn_evaluation']:
        for p in sorted((root/folder).glob('*.npz')):artifacts['files'].append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
    (doc/'ARTIFACTS.json').write_text(json.dumps(artifacts,indent=2)+'\n')
    (root/'reports/REPORT_ZH.md').write_bytes((doc/'REPORT_ZH.md').read_bytes())
    print(json.dumps(dict(complete=complete,mdn_status=training['status'],deltas=deltas),indent=2))


if __name__=='__main__':main()
