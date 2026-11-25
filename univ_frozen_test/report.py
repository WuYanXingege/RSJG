"""Publish verified anonymous native test statistics, without loading truth."""
import shutil
from bcep.contracts import digest
from bcep_joint.common import read, write
from joint_flow_fourfold.report import validate_completed_fold, public_metadata, csvwrite

FOLDS = ('univ',)
GROUPS = ('all','N_ge_9')
METRICS = ('JADE','JFDE','minADE','minFDE')


def fmt(value):
    return f"{value['mean']:.9f} ± {value['std']:.9f}"


def build_report(run,destination):
    destination.mkdir(parents=True,exist_ok=True)
    config=read(run/'RUN_CONFIG.json');frozen=read(run/'FROZEN_MODELS.json')
    all_seeds=[];all_windows=[];rows=[];decisions={}
    overview=['# UNIV：冻结 A/B 五推理种子测试',
        '本报告使用完整官方heldout_confirmation。新模型B与同折固定A比较；每个推理种子单独生成K=20的完整12帧轨迹，随后汇总五个种子的均值和样本标准差（ddof=1）。训练更新为0。',
        '根据2026-11-25用户指示，已完成训练的UNIV立即执行测试，不等待ETH选模结束。原四折控制器、其训练配置与选模规则保持原状态。测试结果不用于调整其余折训练或重新选择本次权重。']
    for fold in FOLDS:
        folder=run/fold/'test';out=destination/fold;out.mkdir(exist_ok=True)
        seeds,windows,groups,receipt=validate_completed_fold(folder,fold)
        assert read(run/fold/'TEST_RESULT.json')==receipt
        assert frozen['freeze_unix']<receipt['seal_unix']<receipt['first_truth_read_unix']
        state=read(folder/'FROZEN_MODELS.json')
        assert state['exact_unchanged'] and state['before']==state['after'] and state['optimizer_updates']==0
        all_seeds.extend(seeds);all_windows.extend(windows)
        decisions[fold]=read(folder/'SUMMARY.json')['interpretation']
        a=frozen['folds'][fold]['A_epoch'];b=frozen['folds'][fold]['B_epoch']
        g=groups['all'];dense=groups['N_ge_9'];d=g['B_minus_A']
        message=(f"{fold.upper()}：全体JADE变化{100*d['JADE']['mean']:+.6f} cm，JFDE变化{100*d['JFDE']['mean']:+.6f} cm；"
                 f"边缘minADE变化{100*d['minADE']['mean']:+.6f} cm，边缘minFDE变化{100*d['minFDE']['mean']:+.6f} cm（均为B−A）。")
        if dense['available']:
            message+=f" N≥9的JFDE变化{100*dense['B_minus_A']['JFDE']['mean']:+.6f} cm。"
        else:message+=' 测试集无N≥9窗口，密集组为N/A。'
        overview.append(message)
        lines=[f'# {fold.upper()}冻结测试',message,
            f'固定A为第{a}轮、B为第{b}轮；来自开发集既定rank1，不根据测试结果选模。全体{g["scene_count"]}窗、{g["agent_count"]}人次；N≥9为{dense["scene_count"]}窗、{dense["agent_count"]}人次。',
            '以下为五个推理种子2035–2039的均值±样本标准差。A、B单位米；配对差B−A单位厘米，负值表示B误差更低。配对差标准差由五个同种子差值直接计算。',
            '| 分组 | 指标 | A（m） | B（m） | B−A（cm） |','|---|---|---:|---:|---:|']
        for group in GROUPS:
            val=groups[group]
            for metric in METRICS:
                row=dict(fold=fold,group=group,windows=val['scene_count'],agents=val['agent_count'],metric=metric,seed_count=5,ddof=1)
                for branch in ('A','B'):
                    for stat in ('mean','std'):
                        row[f'{branch}_{stat}_m']=val[branch][metric][stat] if val['available'] else None
                for stat in ('mean','std'):
                    row[f'B_minus_A_{stat}_cm']=100*val['B_minus_A'][metric][stat] if val['available'] else None
                rows.append(row)
                if val['available']:
                    change={k:100*v for k,v in val['B_minus_A'][metric].items()}
                    lines.append(f"| {group} | {metric} | {fmt(val['A'][metric])} | {fmt(val['B'][metric])} | {fmt(change)} |")
                else:lines.append(f'| {group} | {metric} | N/A | N/A | N/A |')
        lines+=['每个种子使用同一批测试窗；联合JADE/JFDE按窗等权，边缘minADE/minFDE按有效行人人次加权。五个种子衡量固定权重的推理波动，不是五次独立训练。未把五个种子的样本合并为K=100。',
            'A走原生coordinate/H0入口；B走原生256→20整组压缩、一次冻结修正头、六层桥20步DDIM，eta=0。个人噪声、共享边噪声与桥噪声均显式使用对应推理种子。关闭1 cm校准、整组交换和A轨迹重排。',
            '本折五个种子A/B的全部预测先封存并逐文件核验，再由评分端读取测试未来。原生hard_metric_statistics计算FP32误差，MetricAccumulator累计；推理前后冻结模型状态完全一致。',
            '[逐种子CSV](PER_SEED.csv) · [匿名逐窗CSV](PER_WINDOW.csv) · [评分收据](SCORE_RECEIPT.json) · [模型身份](MODEL_IDENTITY.json)']
        for name in ('PER_SEED.csv','PER_WINDOW.csv','NATIVE_METRICS.json','SUMMARY.json','SCORE_RECEIPT.json','PRE_SCORE_SEAL_VERIFICATION.json','FROZEN_MODELS.json','RUN_CONFIG.json'):
            if name.endswith('.csv'):shutil.copyfile(folder/name,out/name)
            else:write(out/name,public_metadata(read(folder/name)))
        write(out/'MODEL_IDENTITY.json',public_metadata(dict(upstream=read(run/fold/'UPSTREAM_FROZEN.json'),B=read(run/fold/'B_FROZEN.json'))))
        # Public seal omits canonical data indices and host prediction paths.
        seal=read(folder/'PREDICTIONS_FROZEN.json')
        write(out/'PREDICTIONS_FROZEN_PUBLIC.json',dict(fold=fold,original_manifest_sha256=digest(folder/'PREDICTIONS_FROZEN.json'),
            seal_unix=seal['seal_unix'],heldout_future_reads=seal['heldout_future_reads'],
            rows=[{k:r[k] for k in ('seed','window','n','sha256')} for r in seal['rows']]))
        (out/'README.md').write_text('\n\n'.join(lines).replace('|\n\n|','|\n|')+'\n')
    overview+=['| 折 | A固定轮次 | B固定轮次 | 全体窗数／人次 | N≥9窗数／人次 |',
        '|---|---:|---:|---:|---:|']
    for fold in FOLDS:
        v=config['models'][fold]
        overview.append(f"| {fold.upper()} | {v['A_epoch']} | {v['B_epoch']} | {v['windows']} / {v['agents']} | {v['dense_windows']} / {v['dense_agents']} |")
    overview+=['本次只给出UNIV的冻结测试结论，五折确认需等待ETH完成。',
        '五个种子描述固定权重下的推理波动。所有训练配置、模型权重和开发集选模结果在测试前固定；没有训练更新、测试调参或测试选模。',
        '[UNIV中文报告](univ/README.md) · [汇总CSV](SUMMARY.csv) · [逐种子CSV](PER_SEED.csv) · [匿名逐窗CSV](PER_WINDOW.csv) · [协议顺序调整](AUTHORIZATION_AMENDMENT.json)',
        '权重、候选库与原始预测留在训练主机。使用独立输出目录，原四折控制器的源码、配置和工作目录未修改。']
    csvwrite(destination/'SUMMARY.csv',rows);csvwrite(destination/'PER_SEED.csv',all_seeds);csvwrite(destination/'PER_WINDOW.csv',all_windows)
    for name in ('RUN_CONFIG.json','FROZEN_MODELS.json','AUTHORIZATION_AMENDMENT.json'):
        write(destination/name,public_metadata(read(run/name)))
    write(destination/'INTERPRETATION.json',dict(scope='UNIV heldout_confirmation; five inference seeds at fixed weights',folds=decisions))
    (destination/'README.md').write_text('\n\n'.join(overview).replace('|\n\n|','|\n|')+'\n')
    for path in destination.rglob('*'):
        if path.is_file():assert path.suffix in ('.md','.json','.csv') and b'/media/ee615' not in path.read_bytes()
    write(destination/'ARTIFACT_SHA256.json',{str(p.relative_to(destination)):digest(p) for p in sorted(destination.rglob('*')) if p.is_file() and p.name!='ARTIFACT_SHA256.json'})
