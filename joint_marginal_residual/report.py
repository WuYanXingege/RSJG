"""Public anonymous dev outcomes and measured stage timing; no test access."""
import shutil
from pathlib import Path
from bcep.contracts import digest
from bcep_joint.common import read,write
from joint_flow_fourfold.report import public_metadata,csvwrite
from .assets import RUN,SOURCE,FOUR

DOC=SOURCE/'docs/research/joint_marginal_residual/20261124'
Q=('minADE','minFDE','JADE','JFDE')

def build_report(results,config):
 DOC.mkdir(parents=True,exist_ok=True);summary=[];criteria=[];all_windows=[]
 passed=results['zara1']['gate']['passed']
 lines=['# 联合与边缘轨迹修正：固定开发集实验与原生提速',
  '**ZARA1研发门槛：'+('通过。已按条件执行扩折与机制对照。**' if passed else '未通过，按预定规则停止本模型路线。**'),
  '仅使用每折自己的train/selection_dev，种子3201，固定30轮、同一配方。此前已看过ZARA2/ZARA1/HOTEL的旧测试成绩；本实验没有读取测试集，没有利用测试成绩调参或选模。',
  'B*保留冻结Flow、依赖生成、原修正头与六层桥，只在一次原生B解码之后添加共享/去均值个体轨迹位移头。新增27,332参数；M256、K20、12帧、20步DDIM、eta=0。个人和联合场景ID不进入可学习输入。',
  '训练候选流2036、开发流2035。训练归一化常数只由该折全部train的冻结B四项原生指标计算。每逻辑批32窗，跨物理微批累计正确分子分母后才计算四项相对差的mean+max。小头训练不再运行DDIM。']
 for name,result in results.items():
  fold='zara1' if name=='zara1_control' else name;variant='control' if name.endswith('_control') else 'head'
  src=RUN/fold/variant;out=DOC/name;out.mkdir(exist_ok=True)
  epoch=result['rank1_epoch'];gate=result['gate']
  lines.append(f"## {name.upper()}：第{epoch}轮，{'通过' if gate['passed'] else '未通过'}")
  lines.append('最难满足的条件：'+', '.join(gate['hardest_conditions'])+f"；最大超限量{gate['max_violation_m']:.12g} m。未使用不同轮次拼接指标。")
  lines+=['| 分组 | 窗数 | 人次 | 指标 | B（m） | B*（m） | B*−B（cm） |','|---|---:|---:|---|---:|---:|---:|']
  for group in ('all','N_ge_9'):
   b=result['baseline'][group];s=result['metrics'][group]
   for q in Q:
    r=dict(fold=fold,variant=variant,epoch=epoch,group=group,metric=q,windows=0 if b is None else b['scene_count'],agents=0 if b is None else b['agent_count'],
     B=None if b is None else b[q],student=None if s is None else s[q],student_minus_B_cm=None if b is None else 100*(s[q]-b[q]))
    summary.append(r)
    if b is None:lines.append(f'| {group} | 0 | 0 | {q} | N/A | N/A | N/A |')
    else:lines.append(f"| {group} | {b['scene_count']} | {b['agent_count']} | {q} | {b[q]:.12f} | {s[q]:.12f} | {100*(s[q]-b[q]):+.9f} |")
  for c in gate['checks']:
   criteria.append(dict(fold=fold,variant=variant,epoch=epoch,group=c['group'],metric=c['metric'],available=c['available'],passed=c['passed'],
    difference_m=c.get('difference_m'),threshold_m=c.get('threshold_m'),strict=c.get('strict'),margin_m=c.get('margin_m')))
  for file in ('RESULT.json','REPLAY.json','RUN_CONFIG.json','TOP5.json','EPOCH_METRICS.json','PREFLIGHT.json'):
   write(out/file,public_metadata(read(src/file)))
  shutil.copyfile(src/'PER_WINDOW.csv',out/'PER_WINDOW.csv')
  for p in sorted(src.glob('epoch_*_windows.csv')):shutil.copyfile(p,out/p.name)
  for p in ('IDENTITY.json','CONFIG.json','TRAIN_SCALES.json','BASELINE_METRICS.json','TIMING.json'):
   write(out/('CACHE_'+p),public_metadata(read(RUN/fold/'cache'/p)))
  replay=read(src/'REPLAY.json');cache=read(RUN/fold/'cache/TIMING.json');t=result['timing']
  lines.append(f"完整30轮更新{t['train_update_seconds']:.3f}s，验证{t['validation_seconds']:.3f}s，冷缓存阶段已观测墙钟{cache['wall_seconds']:.3f}s，缓存载入{t['cache_load_seconds']:.3f}s；真实观测全dev回放{replay['wall_seconds']:.3f}s。线上覆盖摘要与小头额外平均{replay['online_extra_mean_seconds']*1000:.3f}ms/窗。回放原始联合ID、B轨迹、B*轨迹和原生四项分组指标均与缓存精确一致。")
  lines.append('保留全部30个完整epoch、按该路线同一选择器排序的Top5、best与last；断点包含优化器、RNG、随机窗口顺序、游标、完成轮次和选择状态。')
 if 'zara1_control' in results:
  f=results['zara1'];c=results['zara1_control'];mechanism=[]
  for group in ('all','N_ge_9'):
   for q in Q:mechanism.append(dict(group=group,metric=q,full=f['metrics'][group][q],control=c['metrics'][group][q],full_minus_control_cm=100*(f['metrics'][group][q]-c['metrics'][group][q])))
  csvwrite(DOC/'MECHANISM.csv',mechanism)
  lines+=['## 等参数机制对照',
   '对照与完整头使用相同初始化、参数量、教师缓存、四项损失、30轮预算和同一选模规则，仅令P=J、a=0。[实际配对差](MECHANISM.csv)。负差表示完整双分配信息更好；若完整头与对照均有效但完整头无额外优势，只支持轨迹修正有效，不支持该机制信息有额外作用。']
 bench=read(RUN/'benchmark/RESULT.json')
 lines+=['## 四阶段与全链计时',
  '以下是同机同设备、同数据与随机数的真实代表性微基准，至少三次预热后配对中位数。此轮允许与原主线共享GPU，逐次资源占用记录在benchmark中；不把资源竞争归因于优化。不重训旧基线，也不把旧B验证除以小头更新声称全网加速。',
  '| 阶段 | 部分 | 原始中位数(s) | 当前中位数(s) | 比值（原始/当前） |','|---|---|---:|---:|---:|']
 for r in bench['timings']:lines.append(f"| {r['stage']} | {r['component']} | {r['baseline_median_seconds']:.6f} | {r['optimized_median_seconds']:.6f} | {r['speed_ratio']:.4f} |")
 historical={};estimated={};mapping={'density':'upstream/density','head':'upstream/head','coordinate':'upstream/joint','dependency':'dependency/dependence'}
 for stage,path in mapping.items():
  history=read(FOUR/'zara1'/path/'EPOCH_METRICS.json')
  old_update=sum(r['train_seconds'] for r in history);old_val=sum(r['validation_seconds'] for r in history)
  ratio={r['component']:r['speed_ratio'] for r in bench['timings'] if r['stage']==stage}
  historical[stage]=dict(recorded_update_seconds=old_update,recorded_validation_seconds=old_val,completed_epochs=len(history))
  estimated[stage]=old_update/ratio['update_seconds']+old_val/ratio['validation_seconds']
 write(DOC/'FULL_CHAIN_ESTIMATE.json',dict(historical_measured_segments=historical,estimated_full_training_seconds_by_stage=estimated,
  estimated_total_seconds=sum(estimated.values()),full_optimized_training_actually_run=False,
  assumption='representative measured stage ratios apply to historical epoch workload; shared-GPU contention makes this a rough estimate, not measured full-run speedup'))
 lines.append(f"旧四阶段历史更新+验证合计{sum(v['recorded_update_seconds']+v['recorded_validation_seconds'] for v in historical.values())/3600:.3f}h；套用本次代表性分段比值后的全程**估算**为{sum(estimated.values())/3600:.3f}h。没有完整重跑优化后的旧四阶段，因此这一数值不是实测全网墙钟。此次实际复用上游权重，省去density/旧head/coordinate/B依赖重训；本次支付缓存、小头30轮、开发回放和微基准开销。")
 lines+=['保留原Flow、依赖、旧修正头和六层桥。新路径不调用A专用KMeans/H0、旧交换、重排或1cm校准。冻结后才允许跨epoch缓存完整B轨迹；权重变化阶段只允许静态数据/冻结编码与独立随机输入缓存。原有density静态LRU保留；旧head的未增强dev编码可复用，随机增强train编码不冒充同一特征。桥全量地图特征缓存没有盲目启用（ZARA1 dev可达约40GiB），未宣称其已提速。',
  '原生可控优化包括块内modulation、rollout内proposal embedding、schedule、监督边/真值速度、Hungarian传输和日志归约，以及旧head开发编码有界缓存。未通过数值/收益验收的优化应关闭；具体实际执行策略与无明显收益的阶段见benchmark结果，所有不利比值如实保留。',
  '完整缓存、标签、原始观测、权重、完整轨迹与行人ID留在训练主机。公开结果只含匿名窗号、计数和指标。后续冻结测试需另行固定配置，本报告不包含新测试分数。',
  '[汇总CSV](SUMMARY.csv) · [逐条件判定](CRITERIA.csv) · [逐阶段计时](benchmark/TIMINGS.csv) · [运行配置](RUN_CONFIG.json)']
 csvwrite(DOC/'SUMMARY.csv',summary);csvwrite(DOC/'CRITERIA.csv',criteria)
 write(DOC/'RUN_CONFIG.json',public_metadata(config));write(DOC/'DECISION.json',dict(zara1_pass=passed,trained=list(results),gates={k:v['gate'] for k,v in results.items()},test_access=0))
 write(DOC/'STAGE_WALL_TIMES.json',read(RUN/'STAGE_WALL_TIMES.json'))
 (DOC/'benchmark').mkdir(exist_ok=True)
 for p in (RUN/'benchmark').rglob('*'):
  if p.is_file() and p.suffix in ('.json','.csv'):
   target=DOC/'benchmark'/p.relative_to(RUN/'benchmark');target.parent.mkdir(parents=True,exist_ok=True)
   if p.suffix=='.json':write(target,public_metadata(read(p)))
   else:shutil.copyfile(p,target)
 (DOC/'README.md').write_text('\n\n'.join(lines).replace('|\n\n|','|\n|')+'\n')
 write(DOC/'ARTIFACT_SHA256.json',{str(p.relative_to(DOC)):digest(p) for p in sorted(DOC.rglob('*')) if p.is_file() and p.name!='ARTIFACT_SHA256.json'})
