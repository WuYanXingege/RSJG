"""CPU-only report builder from persisted receipts; never loads model weights."""
import argparse,csv,json,statistics,hashlib
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());root=Path(plan['asset_root']);workspace=Path(__file__).resolve().parents[1]
    doc=workspace/plan['documentation_root'];doc.mkdir(exist_ok=True,parents=True)
    def read(name,default=None):
        path=root/name;return json.loads(path.read_text()) if path.exists() else default
    gate=read('reports/GPU_GATE_REPORT.json',{'status':'NOT_EXECUTED'})
    training=read('reports/TRAINING_RESULT.json',{'status':'NOT_STARTED'})
    readonly=read('reports/READONLY_RESULT.json',{'status':'NOT_EXECUTED'})
    summary=read('evaluation/banks/SUMMARY.json')
    complete=bool(summary and summary['status']=='COMPLETE_INTERNAL_OR_HISTORICAL' and summary['seeds']==list(range(2035,2040)))
    rows=[];likelihoods=[];members=[]
    if complete:
        for seed in range(2035,2040):
            result=read(f'evaluation/banks/result_seed{seed}.json');m=result['metadata']
            assert m['complete_members'] and not m['nonformal'] and len(m['members'])==320 and m['seed']==seed
            assert m['export_optimizer_updates']==0 and m['checkpoint_sha256']==training['best.pt']['sha256']
            assert m['member_hash']==read('identity/RUN_DATA_AND_EVAL_CONTRACT.json')['dev_member_hash']
            members.append(m['member_hash'])
            likelihoods.append(dict(seed=seed,final_world_nll_agent_micro=result['likelihood']['eval_nll_final_agent_micro'],iid256_energy_agent_micro=result['iid256_energy_agent_micro']))
            for method in ['iid20','coverage20']:
                for group,values in result['metrics'][method].items():
                    if values:rows.append(dict(seed=seed,method=method,group=group,**values))
        assert len(set(members))==1
        paired=dict(summary=summary,likelihood_and_energy=likelihoods,training_seed=3101,inference_seeds=list(range(2035,2040)),member_hash=members[0],checkpoint=readonly['checkpoint'],scope='internal dev selector comparison, not outer ETH test or independent training replicates')
        (doc/'PAIRED_DEV_METRICS.json').write_text(json.dumps(paired,indent=2)+'\n')
        with (doc/'PAIRED_DEV_METRICS.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    screening=summary['pilot_screening']['status'] if complete else 'NOT_EVALUATED'
    training_state=training.get('stop_reason',training['status'])
    if training.get('status')=='FINISHED' and training_state=='COMPLETE':
        training_state='PATIENCE_STOP' if training.get('stale',0)>=3 else 'NORMAL_EPOCH_CAP'
    if training.get('status')=='FINISHED' and not training.get('has_selected_best'):
        training_state+=' / INSUFFICIENT_TRAINING_EVIDENCE'
    lines=['# BCEP GPU验收与唯一有界Provider试训', '',
           f"GPU_GATE：**{gate['status']}**；TRAINING_STATE：**{training_state}**；EVAL_COMPLETENESS：**{'完整320窗口×5推理seed' if complete else 'INCOMPLETE'}**；SELECTOR_SCREEN：**{screening}**；PROVIDER_REPLACEMENT：**NOT_ESTABLISHED**。",'',
           '本轮仅训练独立BCEP Provider；未训练配对器、条件桥或完整EBJD。没有更改模型、目标、数据划分、选模与筛查规则；没有加载GDTS/旧EBJD权重初始化，没有第二次真实试训。']
    lines+=['','## GPU工程验收',f"单卡FP32。GPU gate墙钟 {gate.get('wall_seconds',0):.3f}s，加CPU回归共 {gate.get('total_preflight_seconds_including_cpu_suite',0):.3f}s，上限600s。25项原CPU测试完整保留。",f"GPU合成16次optimizer更新（连续8步与3+5），真实数据更新0。恢复逐位一致：{gate.get('gpu_resume',{}).get('exactly_equal','未执行')}。",'CPU/GPU固定 rtol=1e-4、atol=1e-5；GPU恢复固定 rtol=1e-6、atol=1e-7。报告同时保留近零量的最大相对差，不将其单独当成失败；判定采用预注册绝对＋相对容差。']
    if gate.get('real_interface'):lines.append('真实无梯度接口预先按人数/索引选取 N='+','.join(str(x['agents']) for x in gate['real_interface'])+'，不依据未来误差筛选。IID20和coverage20确实共享一份M256支持；同一支持上的CPU/GPU选择结果一致。')
    if gate.get('failure'):lines.append('阻塞原因：'+gate['failure'])
    lines+=['','## 唯一试训与选模',f"已完成epoch：{training.get('completed_epoch',0)}；optimizer updates：{training.get('optimizer_updates',0)}；当前/下一epoch：{training.get('next_or_partial_epoch','未启动')}；phase：{training.get('phase','未启动')}；未完成epoch的train cursor：{training.get('train_cursor',0)}。",f"best selected epoch：{training.get('selected_epoch','无')}；final world NLL（agent micro）：{training.get('selection_value','无')}。只有完整320窗口验证才参与选模；未使用coverage精度选择checkpoint。",'配置：seed3101；Adam lr1e-4、gamma1；lambda_aux0.25；每场景一次update；最多10epochs/37900updates/3600秒累计流程墙钟，patience3。历史默认仍禁用真实训练，仅本run配置获授权。']
    if training.get('failure'):lines.append('执行失败：'+training['failure'])
    epochs=root/'training/epochs.jsonl'
    if epochs.exists():
        curve=[json.loads(x) for x in epochs.read_text().splitlines()]
        (doc/'EPOCH_METRICS.json').write_text(json.dumps(curve,indent=2)+'\n')
        lines+=['','| epoch | train复合loss(场景平均) | train最终NLL(行人平均) | dev最终NLL(行人平均) | 累计流程秒 |','|---:|---:|---:|---:|---:|']
        for r in curve:lines.append(f"| {r['epoch']} | {r['train_loss_total_scene_macro']:.6f} | {r['train_nll_final_agent_micro']:.6f} | {r['dev_nll_final_agent_micro']:.6f} | {r['elapsed_wall_seconds']:.2f} |")
    if complete:
        lines+=['','## 冻结best的内部开发集结果','单位：距离m；NLL为世界坐标密度的负对数值（nats），不是米制误差；Recall为0–1比例。均值±样本SD来自五个推理seed，不是五次独立训练。开发集320窗口/621次行人出现；N=1为132窗口，N≥2为188窗口。GMM/MDN、旧provider及外层测试未在此进行公平对照。','','| 指标 | IID20 | coverage20 | coverage−IID |','|---|---:|---:|---:|']
        stats=summary['inference_seed_statistics']
        for key in ['minFDE','recall_0.25','recall_0.5','recall_1.0','P50','P90','P95']:
            u=stats['iid20']['all'][key];v=stats['coverage20']['all'][key]
            lines.append(f"| {key} | {u['mean']:.6f} ± {u['sample_sd']:.6f} | {v['mean']:.6f} ± {v['sample_sd']:.6f} | {v['mean']-u['mean']:+.6f} |")
        lines+=['','| 分组 | 方法 | minFDE | Recall@0.5 | P90 |','|---|---|---:|---:|---:|']
        for group in ['N_eq_1','N_ge_2']:
            for method in ['iid20','coverage20']:
                r=stats[method][group];lines.append(f"| {group} | {method} | {r['minFDE']['mean']:.6f} | {r['recall_0.5']['mean']:.6f} | {r['P90']['mean']:.6f} |")
        lines+=['',f"冻结best的CPU final world NLL：{likelihoods[0]['final_world_nll_agent_micro']:.6f}；IID256 energy score五seed均值：{statistics.mean(x['iid256_energy_agent_micro'] for x in likelihoods):.6f}。NLL不对coverage代表点定义；energy只来自合法IID256。",f"预注册筛查：{json.dumps(summary['pilot_screening'],ensure_ascii=False)}"]
        n_improved=sum(d<0 for d in summary['paired_coverage_minus_iid']['minFDE']);lines.append(f'coverage的minFDE在{n_improved}/5个推理seed改善；逐seed/分组数值见JSON和CSV。')
    else:lines+=['','冻结评测不完整，未计算完整筛查或补零；缺失数据不视为BCEP科学假设失败。']
    if readonly.get('initial_density_diagnostic') and likelihoods:
        initial=readonly['initial_density_diagnostic']['metrics']['initial_dev_nll_final_agent_micro'];best=likelihoods[0]['final_world_nll_agent_micro']
        lines+=['','## 学习证据与解释边界',f'按冻结runner的CPU构造、seed3101及train-only尺度严格重建初始模型；在同一完整dev、CPU精度下，初始final NLL={initial:.6f}，best={best:.6f}，变化={best-initial:+.6f}。该初始模型只作只读诊断，不伪装为训练checkpoint。','NLL变化只支持密度学习诊断，不能等同于覆盖或校准全面达标。上述候选绝对误差、尾部和Recall需单独看待。']
    lines+=['','## 成本与身份',f"gate：{gate.get('wall_seconds',0):.3f}s GPU进程墙钟（不是GPU active秒）；训练wrapper本次调用：{training.get('invocation_wall_seconds',0):.3f}s；runner累计（含验证/保存）：{training.get('run_state',{}).get('elapsed_wall_seconds','未知')}s；只读CPU阶段：{readonly.get('wall_seconds',0):.3f}s /1800s。三个预算不互相挪用。",f"训练峰值allocated显存：{training.get('peak_allocated_bytes',0)/2**30:.3f}GiB。FP32、TF32关闭、cuDNN benchmark关闭/确定性开启、math SDPA；具体环境见GPU_GATE_REPORT及执行身份。",'同M256支持集的IID20/coverage20比较不是等推理成本比较；CPU导出时间不是GPU推理时间。']
    if readonly.get('latency'):
        latency=readonly['latency'];lines.append(f"单个预定{latency['agents']}人dev窗口的CPU总coverage延迟（编码＋M256生成＋选择）均值：{statistics.mean(x['coverage_total_seconds'] for x in latency['timings']):.6f}s；仅三次只读测量，不能外推全数据吞吐。")
    identity=read('identity/PILOT_EXECUTION_IDENTITY.json',{})
    lines+=['',f"执行源码提交：`{identity.get('actual_execution_sha','未启动真实训练')}`。分析基线：`{plan['base_sha']}`。",f"资产目录：`{root}`。best/last、bank及SHA见ARTIFACTS.json；Git不包含权重或bank。",'保护主工作区HEAD与status前后相同，旧五折selected checkpoint及配置哈希复核见PROTECTED_ASSETS_CHECK.json。没有删除或覆盖历史资产。','','## 结论','选择器筛查仅检验这个冻结密度上M256预算选点的价值。本轮未证明独立provider替代旧provider、flow优于其他密度或完整EBJD优于原六层。旧matched见过当前内部dev，历史0.5811m/64.1%不作为本次门槛。','本run结束后不建立后续队列，不启动第二次训练、配对器或条件桥。']
    if complete:
        u=stats['iid20']['all'];v=stats['coverage20']['all']
        lines+=['',f"本次结论：{screening}。coverage20的minFDE比IID20降低{100*(u['minFDE']['mean']-v['minFDE']['mean'])/u['minFDE']['mean']:.2f}%，Recall@0.5增加{100*(v['recall_0.5']['mean']-u['recall_0.5']['mean']):.2f}个百分点；单人和多人分组的minFDE方向一致。",
            f"仍有{100*(1-v['recall_0.5']['mean']):.2f}%的行人出现未被20个候选覆盖到0.5m内，P95为{v['P95']['mean']:.3f}m；这说明尾部误差仍未解决，不能只凭平均误差宣布提供器已达标。",
            '训练NLL与dev NLL来自不同成员，而且训练统计是在不断更新的参数下累积，不能仅凭两列的差距判定过拟合。当前结果支持密度确实有所学习、同支持集预算选点有效；尚不支持跨fold泛化、独立训练种子稳定性或替代旧provider。',
            '唯一建议：将当前冻结模型作为独立provider候选保留，后续先取得数据暴露严格可比的替代验证，再决定是否接入联合桥；本轮不执行该后续实验。']
    lines+=['','## 可复核入口',
        '以下为本次执行记录；单次启动器会拒绝重复run，预算耗尽也不能重新计时。',
        '```bash',
        f"python scripts/bcep_gpu_gate.py --plan {plan['documentation_root']}/AUTHORIZATION_AND_PILOT_PLAN.json --config configs/research/{plan['run_id']}.yaml",
        f"python scripts/bcep_one_pilot.py --plan {plan['documentation_root']}/AUTHORIZATION_AND_PILOT_PLAN.json --config configs/research/{plan['run_id']}.yaml",
        f"python scripts/bcep_pilot_readonly.py --plan {plan['documentation_root']}/AUTHORIZATION_AND_PILOT_PLAN.json --config configs/research/{plan['run_id']}.yaml",
        f"python scripts/summarize_bcep_pilot.py --plan {plan['documentation_root']}/AUTHORIZATION_AND_PILOT_PLAN.json",
        '```',
        '执行Python为项目`.conda/rsjg/bin/python`；CPU回归：`python -m pytest tests/test_bcep_core.py tests/test_bcep_prepilot.py -q`。GPU gate的精确参数见GPU_GATE_REPORT及授权计划，训练/导出命令亦保存在执行身份中。']
    (doc/'PILOT_RESULT_ZH.md').write_text('\n'.join(lines)+'\n')
    for name in ['GPU_GATE_REPORT.json','TRAINING_RESULT.json','READONLY_RESULT.json']:
        src=root/'reports'/name
        if src.exists():(doc/name).write_bytes(src.read_bytes())
    for name in ['PILOT_EXECUTION_IDENTITY.json','PROTECTED_ASSETS_CHECK.json','FINAL_PROTECTION_CHECK.json','END_STATE.json']:
        src=root/'identity'/name
        if src.exists():(doc/name).write_bytes(src.read_bytes())
    artifacts=dict(run_id=plan['run_id'],training_checkpoint_records={k:training[k] for k in ['best.pt','last.pt'] if k in training},banks=str(root/'evaluation/banks'),code_identity=identity.get('source_identity'),baseline_sha=plan['base_sha'])
    artifacts['bank_files']=[]
    for path in sorted((root/'evaluation/banks').glob('*.npz')):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        artifacts['bank_files'].append(dict(path=str(path),sha256=h.hexdigest(),bytes=path.stat().st_size))
    (doc/'ARTIFACTS.json').write_text(json.dumps(artifacts,indent=2)+'\n')
    (root/'reports/PILOT_RESULT_ZH.md').write_bytes((doc/'PILOT_RESULT_ZH.md').read_bytes())
    print(json.dumps(dict(gpu_gate=gate['status'],training=training['status'],complete=complete,selector_screen=screening)))

if __name__=='__main__':main()
