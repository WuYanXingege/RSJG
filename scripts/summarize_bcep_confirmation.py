"""Render completed/partial confirmation evidence without launching computation."""
import argparse,csv,json,shutil,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bcep_confirmation.common import read,write

def main(plan_path):
    plan=read(plan_path);root=Path(plan['assets']);docs=Path(plan['docs']);status=read(root/'PIPELINE_STATE.json')
    assert not status.get('active',True),'wait for bounded task termination'
    split=read(plan['split_lock']);contract=read(plan['contract']);runs=read(root/'reports/TRAINING_RECEIPTS.json') if (root/'reports/TRAINING_RECEIPTS.json').exists() else []
    protection=read(root/'reports/PROTECTION_RECEIPT.json');complete=(root/'reports/AGGREGATES.json').exists()
    agg=read(root/'reports/AGGREGATES.json') if complete else None
    cpu=read(root/'receipts/CPU_PREPARATION_BUDGET.json');gpu=read(root/'receipts/GPU_PREFLIGHT_BUDGET.json')
    evaluation=read(root/'receipts/READONLY_CPU_BUDGET.json') if (root/'receipts/READONLY_CPU_BUDGET.json').exists() else {'seconds':0}
    text=['# BCEP 新录制序列确认与独立训练重复','',
          f"执行状态：**{status['status']}**；确认评测：**{'CONFIRMATION_COMPLETE' if complete else 'EVALUATION_INCOMPLETE'}**。",
          f"本轮确认录制组 **{split['chosen_recording']}**，训练/选模/确认分别 {len(split['train_indices'])}/{len(split['selection_dev_indices'])}/{len(split['heldout_confirmation_indices'])} 窗口。确认集含 {contract['counts']['heldout_confirmation']['agent_occurrences']} 次有效行人出现。",
          '这是四个新初始化模型未训练、未选模的留出录制序列；该序列可能在旧项目中被训练或观察过，不是正式独立外层测试，也不是未见地图/地点测试。','',
          '## 固定数据与执行合同','',
          '- 证据起点：074d54b04be6a717b2c248a1fe3c2596a4fd9be8；使用既有 Flow 和 MDN8，没有新增网络或选择器。',
          f"- 源训练 manifest SHA：`{split['source_sha256']}`；新 split 文件 SHA：`{plan['split_sha256']}`。",
          '- 原 uni_examples 320 窗口保持 selection_dev，仅以最终时刻世界坐标 NLL 的有效行人 micro 平均选模；相同值保留早期 epoch。',
          '- 确认组按合格录制窗口数降序、规范录制 ID 字典序选定；先锁元数据身份，再读取新 train 的 future 拟合尺度。train/selection_dev/heldout_confirmation/excluded 不交且覆盖全部4110个源窗口。',
          '- 七个规范原始注释序列直接经过已有 synchronized_windows 导出，没有在源成员中加入旋转/镜像/增强别名。录制ID与静态地图组分别登记；Zara 三个录制共享语义地图。原视频未另下载取证。来源及别名规则见 SPLIT_LOCK 和 RUNNER_ADAPTATION。',
          '- 新 train 共30375次行人出现用于重新拟合 motion scales；没有复用包含确认组的旧 scale 表。静态地图裁剪只以最后观测位置为中心，没有确认未来热图或教师候选。',
          '- 固定四组顺序：Flow3102 → MDN83102 → MDN83103 → Flow3103。没有重跑3101或加载旧训练权重。',
          '- Adam 1e-4，gamma=1，辅助多horizon权重0.25；一个完整场景/update、FP32。每组最多6轮/16752 updates/3600秒，patience=3；不转移节省预算。',
          '- 同seed共享编码器初值和shuffle；不同seed初值不同。真实初始化记录逐值哈希与CPU预检一致。',
          '- 所有四组先终止并冻结best，再以M256生成IID20和固定KMeans20。KMeans仍为5次重启/100次迭代/1e-6 m、局部SHA种子；不调参、不选择新checkpoint。',
          '- 确认所有模型×五推理seed的候选输出全部落盘冻结后，才允许读取确认future评分。CPU模型前向FP32；KMeans和终点误差统一float64；NLL来自原连续密度，不在代表点上计算。','',
          '## 训练收据','',
          '| 密度/训练seed | 完成epoch | best epoch/update | last update | 终止原因 | 流程墙钟秒 | selection NLL |',
          '|---|---:|---:|---:|---|---:|---:|']
    for r in runs:
        best=r.get('best');epochs=[json.loads(x) for x in (root/'training'/r['name']/'epochs.jsonl').read_text().splitlines()]
        stop=r['stop_reason']
        if r['phase']=='complete':stop='MAX_EPOCHS' if r['completed_epoch']==6 else 'PATIENCE_3'
        text.append(f"| {r['name']} | {r['completed_epoch']} | {str(best['epoch'])+'/'+str(best['update']) if best else '无合格best'} | {r['last_update']} | {stop} | {r['cumulative_wall_seconds']:.3f} | {best['selection_value'] if best else '—'} |")
    text+=['','last checkpoint 的 epoch 字段是下一阶段游标，完整轮数以 completed_epoch 为准；没有把未完成更新计作完整epoch。best和last均保存模型、optimizer、scheduler、游标、RNG与预算，不以任意last冒充best。']
    decision={'engineering_complete':protection['passed'] and len(runs)==4 and all(r.get('best') is not None for r in runs),
              'confirmation_complete':bool(complete),'provider_replacement':'NOT_ESTABLISHED','full_ebjd_performance':'NOT_EVALUATED'}
    paired=[]
    if complete:
        def get(kind,method,seed,metric,group='all'):
            return next(x['metrics'][metric] for x in agg['per_training_seed'] if (x['model_kind'],x['method'],x['training_seed'],x['group'])==(kind,method,seed,group))
        def aggregate(kind,method,metric):
            return next(x['metrics'][metric] for x in agg['across_training_seeds'] if (x['model_kind'],x['method'],x['group'])==(kind,method,'all'))
        metrics=['minFDE','recall_0.25','recall_0.5','recall_1.0','P50','P90','P95']
        text+=['','## 每个独立训练seed的确认指标','',
               '各格为5个推理seed均值±样本SD；误差单位m，Recall为%。不是5次训练重复。逐seed值在 PER_RUN_PER_SEED.csv。','',
               '| 密度 | 训练seed | 选择器 | minFDE | Recall .25 | Recall .5 | Recall 1.0 | P50 | P90 | P95 |',
               '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
        for kind in ['bcep_flow','mdn8']:
            for seed in [3102,3103]:
                for method in ['IID20','KMeans20']:
                    cells=[]
                    for m in metrics:
                        s=get(kind,method,seed,m);factor=100 if m.startswith('recall') else 1
                        cells.append(f"{s['mean']*factor:.6f} ± {s['inference_sd']*factor:.6f}")
                    text.append('| '+' | '.join([kind,str(seed),method,*cells])+' |')
        text+=['','## 两个训练seed等权汇总','',
               '先分别平均5个推理seed，再对3102、3103两个训练均值等权聚合。下面SD仅基于n=2；不是10次独立训练，也不是把窗口当作独立录制。','',
               '| 密度 | 选择器 | minFDE均值±训练seed SD | 两个训练均值/范围 | Recall .5均值% | P90均值 | P95均值 |',
               '|---|---|---:|---|---:|---:|---:|']
        for kind in ['bcep_flow','mdn8']:
            for method in ['IID20','KMeans20']:
                v=aggregate(kind,method,'minFDE');vals=v['individual_training_means']
                text.append(f"| {kind} | {method} | {v['mean']:.6f} ± {v['training_seed_sd']:.6f} | {vals[0]:.6f}, {vals[1]:.6f}；[{min(vals):.6f}, {max(vals):.6f}] | {aggregate(kind,method,'recall_0.5')['mean']*100:.6f} | {aggregate(kind,method,'P90')['mean']:.6f} | {aggregate(kind,method,'P95')['mean']:.6f} |")
        diffs=[]
        text+=['','## 配对方向与选择器迁移','','误差为左−右（负值改善）；Recall差为百分点（正值改善）。','','| 对照 | 训练seed | ΔminFDE | ΔRecall .5 pp | ΔP90 | ΔP95 |','|---|---:|---:|---:|---:|---:|']
        for seed in [3102,3103]:
            for label,left,right in [('Flow−MDN8 KMeans',('bcep_flow','KMeans20'),('mdn8','KMeans20')),('Flow KMeans−IID',('bcep_flow','KMeans20'),('bcep_flow','IID20')),('MDN8 KMeans−IID',('mdn8','KMeans20'),('mdn8','IID20'))]:
                d={m:get(*left,seed,m)['mean']-get(*right,seed,m)['mean'] for m in metrics};paired.append(dict(contrast=label,training_seed=seed,**{(m+'_pp' if m.startswith('recall') else m):(100*v if m.startswith('recall') else v) for m,v in d.items()}))
                text.append(f"| {label} | {seed} | {d['minFDE']:+.6f} | {100*d['recall_0.5']:+.6f} | {d['P90']:+.6f} | {d['P95']:+.6f} |")
                if label.startswith('Flow−MDN8'):diffs.append(d['minFDE'])
        decision['training_seed_direction']='SAME_DIRECTION' if diffs[0]*diffs[1]>0 else 'MIXED_OR_TIED'
        decision['flow_minus_mdn_minFDE']=diffs
        transfer=[r['minFDE'] for r in paired if 'KMeans−IID' in r['contrast']]
        decision['selector_transfer']='IMPROVED_ALL_FOUR_MINFDE' if all(v<0 for v in transfer) else 'MIXED_OR_NOT_IMPROVED'
        for group in ['N1','Nge2']:
            text+=['',f'### {group}（保留原人数，不裁剪）','','| 密度/训练seed | 方法 | minFDE | Recall .5 % | P95 |','|---|---|---:|---:|---:|']
            for kind in ['bcep_flow','mdn8']:
                for seed in [3102,3103]:
                    for method in ['IID20','KMeans20']:
                        text.append(f"| {kind}/{seed} | {method} | {get(kind,method,seed,'minFDE',group)['mean']:.6f} | {100*get(kind,method,seed,'recall_0.5',group)['mean']:.6f} | {get(kind,method,seed,'P95',group)['mean']:.6f} |")
        text+=['','### 连续密度NLL','','| 模型/训练seed | 确认最终NLL（agent micro） |','|---|---:|']
        for r in read(root/'reports/CONFIRMATION_NLL.json'):text.append(f"| {r['run']} | {r['final_world_nll_agent_micro']:.6f} |")
        text+=['','## 判断','',f"TRAINING_SEED_DIRECTION={decision['training_seed_direction']}；Flow−MDN8 的两次 minFDE 差为 {diffs[0]:+.6f}、{diffs[1]:+.6f} m。保留表中尾部和Recall取舍，不根据其中一项挑选训练seed。",f"SELECTOR_TRANSFER={decision['selector_transfer']}；这是固定同支持池 KMeans 相对 IID 的迁移结果，不是新选择器搜索。",
               '该证据只支持本轮录制、两次初始化和已冻结协议下的观察。即便Flow平均误差较好，也没有直接证明替代matched Provider；如果方向不一致，则独立训练随机性仍是实质限制。不能把历史uni_examples/3101表或旧ETH数字并入此表。']
    else:text+=['','确认评测尚不完整。已完成的权重和输出保留；不删场景、不补seed、不换模型以凑完整结论。',status.get('error','')]
    text+=['','## 成本与保护','',f"- CPU准备与预检：{cpu['seconds']:.3f}s/1800s；GPU预检：{gpu['seconds']:.3f}s/600s。预检包含48次CPU合成更新、零正式更新；GPU预检仅合成forward/backward，不创建optimizer。",
           f"- 四run累计训练流程：{sum(r['cumulative_wall_seconds'] for r in runs):.3f}s；各自独立3600s预算，最多一个在途场景及保存超界，不追加第二轮训练。",
           f"- CPU冻结导出/评分：{evaluation['seconds']:.3f}s/7200s。正式真实更新合计 {sum(r['last_update'] for r in runs)}。",
           f"- 保护检查：{len(protection['files'])}项前后哈希，PASS={protection['passed']}；原六层和原BCEP工作区HEAD/status未变。未删除、覆盖旧资产。"]
    if (root/'reports/EVALUATION_TIMING.json').exists():
        times=read(root/'reports/EVALUATION_TIMING.json');text+=['','| 模型 | 5seed数据读取s | 编码器+支持生成s | CPU转移/复制s | KMeans选择s | 完整候选流程s |','|---|---:|---:|---:|---:|---:|']
        for name in [r['name'] for r in runs]:
            rs=[r for r in times if r['run']==name]
            text.append('| '+name+' | '+' | '.join(f"{sum(r[k] for r in rs):.3f}" for k in ['data_seconds','model_encoder_and_support_seconds','cpu_transfer_copy_seconds','kmeans_seconds','complete_candidate_seconds'])+' |')
        text+=['','模型编码器前向与支持采样在既有 sample_support 中联合计时，未伪造独立分项；CPU转移项是本次CPU推理的转移/数组复制，不是CUDA带宽。完整候选流程包含读取和落盘，但不含后续NLL评分。历史27ms选择成本不是本轮端到端实时性能。']
    text+=['','## 结束状态与下一步','',f"ENGINEERING_COMPLETE={decision['engineering_complete']}；CONFIRMATION_COMPLETE={complete}。PROVIDER_REPLACEMENT=NOT_ESTABLISHED；FULL_EBJD_PERFORMANCE=NOT_EVALUATED。",
           '本轮不继续在uni_examples或确认录制上调参。唯一下一步是：依据这次稳定性与取舍，另行限定预算开展固定自有Provider的六层联合pilot；该pilot不属于本轮授权，未自动启动。',
           f"全部大权重和bank位于 `{root}`。默认训练开关仍关闭，无后续队列。固定执行提交见CHECKPOINTS_FROZEN中identity.source_commit；选中权重和输出SHA见同名冻结收据。",'']
    if complete:
        fm=aggregate('bcep_flow','KMeans20','minFDE')['mean'];mm=aggregate('mdn8','KMeans20','minFDE')['mean']
        f95=aggregate('bcep_flow','KMeans20','P95')['mean'];m95=aggregate('mdn8','KMeans20','P95')['mean']
        gains={k:100*(1-aggregate(k,'KMeans20','minFDE')['mean']/aggregate(k,'IID20','minFDE')['mean']) for k in ['bcep_flow','mdn8']}
        decision['mean_flow_minus_mdn_minFDE_m']=fm-mm
        decision['mean_flow_minus_mdn_P95_m']=f95-m95
        decision['kmeans_minFDE_improvement_percent']=gains
        decision['tradeoff']='KMeans improves mean and upper tails but worsens P50 and Recall@0.25 in all four runs; Flow versus MDN minFDE direction differs across training seeds.'
        executive=['## 主要结论','',
            f"**固定KMeans的收益迁移到了本轮留出录制，但Flow优于MDN8的方向没有跨训练seed保持一致。** Flow+KMeans的两训练seed平均minFDE为 **{fm:.6f} m**，MDN8+KMeans为 **{mm:.6f} m**（差 {fm-mm:+.6f} m，平均降低 {100*(mm-fm)/mm:.2f}%）。这不是稳定优势或替代旧Provider的证明。",
            f"- KMeans相对IID的minFDE平均降低：Flow {gains['bcep_flow']:.2f}%，MDN8 {gains['mdn8']:.2f}%；四个run的minFDE、P90、P95及Recall@0.5/1.0均改善，支持固定候选压缩具有实际价值。",
            '- 同时四个run的P50均升高、Recall@0.25均降低：改善主要体现在平均误差及较大误差区间，不能写成所有精度尺度都改善。这里只描述误差分布取舍，不据此推断具体因果机制。',
            f"- Flow−MDN8的minFDE在3102为 {diffs[0]:+.6f} m，在3103为 {diffs[1]:+.6f} m；Flow的P90和Recall@0.5均较好，但P95两次均较差，合并P95差 {f95-m95:+.6f} m。连续密度NLL的优劣同样随训练seed反转。保留Pareto取舍，不挑有利seed。",
            '- 单人组77窗口/77次出现，多人组921窗口/5833次出现；总表主要由多人出现贡献。单人组Flow的minFDE两次均较好，多人组仍呈两seed方向反转，因此人数分组没有消除总体的随机性问题。单人样本量小，不作显著性结论。',
            '- Flow两组训练墙钟各约41–42分钟，MDN8各约16分钟；两者完成相同6轮和16752更新。更高Flow训练成本也应计入方案选择，不能只展示较低的汇总误差。',
            '- 工程与确认流程均完成；尚未建立旧matched Provider的同确认协议对照，也未训练联合桥。因此 PROVIDER_REPLACEMENT=NOT_ESTABLISHED，FULL_EBJD_PERFORMANCE=NOT_EVALUATED。','']
        text[6:6]=executive
    text+=['独立结果复核见 `RESULT_AUDIT.json`；进程退出与GPU空闲核验见 `EXIT_RECEIPT.json`。', '固定执行提交：`ec76719379bdc1b6be73c7476506e6877f715b77`。报告生成脚本仅读取已落盘结果，不进行训练、选模或重新推理。','']
    report='\n'.join(text)
    (docs/'CONFIRMATION_REPORT_ZH.md').write_text(report);(root/'reports/CONFIRMATION_REPORT_ZH.md').write_text(report)
    write(docs/'DECISION.json',decision);write(root/'reports/DECISION.json',decision)
    if paired:
        with (docs/'PAIRED_DELTAS.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(paired[0]),lineterminator='\n');w.writeheader();w.writerows(paired)
        shutil.copyfile(docs/'PAIRED_DELTAS.csv',root/'reports/PAIRED_DELTAS.csv')
    for name in ['GPU_PREFLIGHT.json','CPU_PREFLIGHT.json']:
        if (root/'reports'/name).exists():shutil.copyfile(root/'reports'/name,docs/name)
    for p in (root/'receipts').glob('*_BUDGET.json'):shutil.copyfile(p,docs/p.name)
    print(json.dumps(decision))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);main(p.parse_args().plan)
