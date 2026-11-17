"""CPU-only report from frozen results; never selects a model or launches work."""
import argparse,csv,json,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bcep_joint.common import read,write,digest
METRICS=['minADE','minFDE','JADE','JFDE']
def mean(x):return statistics.mean(x)
def sd(x):return statistics.stdev(x) if len(x)>1 else 0.
def main(plan_path):
    p=read(plan_path);root=Path(p['assets']);docs=Path(p['docs']);out=root/'reports';state=read(root/'PIPELINE_STATE.json');complete=(out/'PER_RUN_PER_SEED.csv').exists() and (out/'EVALUATION_STATUS.json').exists() and read(out/'EVALUATION_STATUS.json')['status']=='COMPLETE'
    budgets={f.name:read(f) for f in (root/'receipts').glob('*BUDGET.json')};write(docs/'BUDGET.json',budgets)
    execution=read(root/'identity/EXECUTION_IDENTITY.json') if (root/'identity/EXECUTION_IDENTITY.json').exists() else {}
    summary=dict(status=state['status'],complete=complete,execution=execution,statistical_unit='two upstream Provider initializations; ONE downstream initialization 3201; five inference seeds per condition; not ten independent trainings',unit='m',budgets=budgets,default_allow_real_training=False)
    text=['# 冻结自有 Flow Provider＋随机初始化六层条件桥联合 pilot','',f"状态：**{state['status']}**。两种上游初始化、同一个下游初始化（3201）；这不是两次独立下游初始化，也不是 Flow/MDN 胜负赛。",'','两种 Flow 均为此前冻结的自有模型；地图 U-Net、共享编码器、六层去噪器和配对头全部随机初始化，未加载 GDTS 或旧 EBJD 学习参数。复用了既有网络实现，并不声称架构元件均为原创。','']
    text+=['## 固定设计与身份','',f"训练执行提交：`{execution.get('execution_commit','未启动')}`。",'',f"split 文件 SHA256：`{p['split_file_sha256']}`；canonical split：`{p['split_hash']}`；source manifest：`{p['source_manifest_sha256']}`。",'', 'train 2792 窗口；selection_dev 为 uni_examples 的 320 窗口；确认集为 ucy/zara02 的 998 窗口、5910 次有效行人出现。共享静态地图，且 ZARA02 成绩此前已公开：这里只是留出录制的后续机制验证，不是首次盲测、未见地点或 ETH 外层测试。','', '训练候选是 Provider 在训练成员上的 in-sample 预测，不是交叉拟合。M=256、K=20、KMeans 5重启/100迭代/1e-6阈值全部冻结；确认候选沿用先前五seed bank，只做一次声明的 float64→float32 转换，不改变中心顺序。KMeans 的 P50/Recall@0.25 取舍仍存在；硬终点桥不能修复候选覆盖缺失。','', '192/4/768、6 FJAB，FULL/CTF/Raw，固定终点 eta=0，20-step DDIM；rollout/projection OFF，FP32。原 joint_pair_loss（权重1、温度0.1）、GT终点条件桥去噪及 coarse/geometry/map 权重0.25/0.05/0.10不变。GT只用于训练监督和冻结后计分，推理API不接真实未来。','', '唯一有意训练调整：所有随机下游参数基础LR统一1e-4；AdamW、原衰减分组、clip5、5轮warmup+30轮cosine时间轴保留。每逻辑微批4场景、累积4，698微批/epoch，含末尾flush共175 optimizer updates/epoch；最大20轮/3500updates，或每run累计7200秒先到即停，余量不转移。显式可恢复随机排列和增强generator保留全场景身份与原损失权重。','', '选模仅用完整 selection_dev 的 H_JADE（seed2035），相同值保留更早完整epoch；不读取确认成绩或旧GDTS eligible阈值。G/H是同一模型的两种推理模式，不是独立训练的无配对消融。','']
    frozen=read(out/'CHECKPOINTS_FROZEN.json') if (out/'CHECKPOINTS_FROZEN.json').exists() else None
    if frozen:
        text+=['## 训练与选模','', '| Provider seed | 完整训练/dev轮数 | updates | best epoch/update | 流程秒 | 状态 |','|---|---:|---:|---|---:|---|']
        for r in frozen['runs']:
            s=r['status'];b=r['best'];text.append(f"| {r['name']} | {s['training_completed_epoch']}/{s['completed_epoch']} | {s['update']} | {str(b['epoch'])+'/'+str(b['update']) if b else '无完整best'} | {r['budget']['seconds']:.2f} | {s['status']} |")
        text+=['','预算包含初始化、训练、完整dev验证和保存。到限后在一个在途 optimizer step/保存边界退出，有限超界单列于BUDGET；未完成dev的last不冒充best。完整best/last路径、SHA、模型/optimizer/scheduler/RNG/游标/累积梯度/选择历史见 CHECKPOINTS_FROZEN.json。','']
    if complete:
        rows=list(csv.DictReader((out/'PER_RUN_PER_SEED.csv').open()));aggregates=[];paired=[];runstats={}
        for run in p['runs']:
            name=run['name'];runstats[name]={}
            for group in ['all','N1','Nge2']:
                r=[x for x in rows if x['run']==name and x['group']==group]
                if not r:continue
                byseed={seed:{} for seed in range(2035,2040)}
                for seed in byseed:
                    for method in ['G','H','H_shuffle']:
                        z=[x for x in r if int(x['inference_seed'])==seed and (x['method'].startswith('H_shuffle_') if method=='H_shuffle' else x['method']==method)]
                        assert len(z)==(5 if method=='H_shuffle' else 1)
                        byseed[seed][method]={k:mean([float(x[k]) for x in z]) for k in METRICS}
                        if method=='H_shuffle':
                            aggregates.append(dict(run=name,group=group,method=method,level=f'seed{seed}_five_permutations',**{k:byseed[seed][method][k] for k in METRICS},**{k+'_sd':sd([float(x[k]) for x in z]) for k in METRICS}))
                for method in ['G','H','H_shuffle']:
                    v={k:mean([byseed[s][method][k] for s in byseed]) for k in METRICS};runstats[name][group,method]=v
                    aggregates.append(dict(run=name,group=group,method=method,level='five_inference_seeds',**v,**{k+'_sd':sd([byseed[s][method][k] for s in byseed]) for k in METRICS}))
                for ref in ['G','H_shuffle']:
                    for k in METRICS:
                        differences=[byseed[s]['H'][k]-byseed[s][ref][k] for s in byseed];rv=mean([byseed[s][ref][k] for s in byseed]);paired.append(dict(run=name,group=group,reference=ref,metric=k,delta_mean=mean(differences),delta_sd=sd(differences),improvement_percent=-100*mean(differences)/rv if rv else None,improved_seeds=sum(x < -1e-6 for x in differences),tied_seeds=sum(abs(x)<=1e-6 for x in differences),worse_seeds=sum(x > 1e-6 for x in differences)))
        names=[r['name'] for r in p['runs']]
        for group in ['all','N1','Nge2']:
            for method in ['G','H','H_shuffle']:
                values=[runstats[n].get((group,method)) for n in names]
                if all(values):aggregates.append(dict(run='equal_two_provider_conditions',group=group,method=method,level='mean_of_two_run_means',**{k:mean([v[k] for v in values]) for k in METRICS},**{k+'_sd':sd([v[k] for v in values]) for k in METRICS}))
        for folder in [out,docs]:
            for file,items in [('SUMMARY_METRICS.csv',aggregates),('PAIRED_DIFFERENCES.csv',paired)]:
                with (folder/file).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(items[0]),lineterminator='\n');w.writeheader();w.writerows(items)
        summary.update(aggregates=aggregates,paired=paired)
        text+=['## 冻结确认结果（米）','', '下表各run为五推理seed均值±样本SD；H_shuffle先对每seed的五个固定纯置换取均值，再跨五seed汇总，置换内SD另存CSV。双Provider条件汇总为两个run均值的等权平均；其SD若查阅CSV表示两条件差异，不是独立下游训练稳定性。','', '| 条件 | 方法 | minADE | minFDE | JADE | JFDE |','|---|---|---:|---:|---:|---:|']
        for r in aggregates:
            if r['group']=='all' and r['level'] in ['five_inference_seeds','mean_of_two_run_means']:
                text.append('| '+r['run']+' | '+r['method']+' | '+' | '.join(f"{r[k]:.6f} ± {r[k+'_sd']:.6f}" for k in METRICS)+' |')
        text+=['','| Provider条件 | 指标 | H−G（负为改善） | 相对改善% | 改善/持平/退化seed |','|---|---|---:|---:|---|']
        for r in paired:
            if r['group']=='all' and r['reference']=='G':text.append(f"| {r['run']} | {r['metric']} | {r['delta_mean']:+.6f} | {r['improvement_percent']:+.3f}% | {r['improved_seeds']}/{r['tied_seeds']}/{r['worse_seeds']} |")
        both_joint=all(runstats[n]['all','H']['JADE']<runstats[n]['all','G']['JADE'] and runstats[n]['all','H']['JFDE']<runstats[n]['all','G']['JFDE'] for n in names)
        both_fde=all(runstats[n]['all','H']['JFDE']<runstats[n]['all','G']['JFDE'] for n in names)
        conclusion='两种Provider条件下JADE/JFDE均改善；minADE代价按上表单列，本轮不事后设阈值宣称全面通过。' if both_joint else ('两种Provider条件下JFDE均改善，但完整路径的联合改善尚未在两条件下一致建立。' if both_fde else 'G→H收益未在两种Provider条件下一致成立；不能挑较好的一次替代总体结论。')
        summary['scientific_conclusion']=conclusion
        text+=['','## 结论与边界','',conclusion,'','H_shuffle是冻结完整轨迹的离线组合破坏对照，不是第三个部署模型或训练消融；是否恶化均如实记录。N=1/N≥2完整结果见SUMMARY_METRICS.csv。固定终点保证G/H minFDE相等；末帧、候选和噪声同步gather、每人双射、endpoint-only JFDE、负控边缘不变均见INVARIANT_RECEIPT。不能仅凭JFDE改善证明桥路径更好，也不能把重叠窗口当独立录制做显著性包装。','', '下一步仅优先判断：本次固定预算内的联合收益是否足以支持独立下游初始化复验；不自动追加训练、改选择器或更换Provider。']
    else:
        text+=['## 未完成项','',f"本轮没有完整确认表，不作机制有效/无效结论。状态与原因：`{json.dumps(state,ensure_ascii=False)}`。",'', '已有权重和已完成输出保留，未自动续训或缩减评测人数/seed。下一步仅先解决记录的具体阻塞；不得借此追加结构实验。']
    text+=['','## 工程检查、成本与复现','', '合成中断恢复覆盖pending梯度、模型、optimizer、scheduler、Python/NumPy/Torch/loader RNG和选择历史；CPU恢复逐值一致。CUDA自适应池化和grid_sample反向无严格确定性内核，因此固定FP32/TF32 OFF并使用确定性warn_only，记录波动来源，不宣称CUDA训练轨迹逐位复现。确认标签在两个best及全部10个G/H轨迹bank冻结后才可用于计分。','', '| 阶段 | 累计秒 | 上限秒 |','|---|---:|---:|']
    for name,b in budgets.items():text.append(f"| {name} | {b.get('seconds',0):.2f} | {b.get('limit_seconds',0)} |")
    text+=['','源码编写时间未计为GPU时间；PREPARATION_CPU是CPU准备墙钟，GPU_PRECHECK/训练/FINAL_EVALUATION是占用GPU任务的流程墙钟（包含其中的CPU/存盘部分）。阶段额度不转移。','',f"资产：`{root}`。训练/评测控制台：`{root/'logs/pilot.log'}`。",'', '复现入口（默认不开训练，必须显式授权，且已执行目录不可重复启动）：','```bash',f"../.conda/rsjg/bin/python -B -u scripts/run_bcep_joint.py --plan {root/'RUN_PLAN.json'} --execute-authorized-two-runs",f"../.conda/rsjg/bin/python -B scripts/report_bcep_joint.py --plan {root/'RUN_PLAN.json'}",'```','', '所有旧资产/工作区HEAD与status的前后保护核对见PROTECTION_RECEIPT；本次结束后没有自动后续队列。']
    for folder in [out,docs]:write(folder/'SUMMARY.json',summary);(folder/'REPORT_ZH.md').write_text('\n'.join(text)+'\n')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);main(parser.parse_args().plan)
