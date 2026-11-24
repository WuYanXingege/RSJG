"""Post-2035 five-seed replication; all adverse outcomes retained."""
import csv
import shutil
from .common import *
from joint_flow_fourfold.report import public_metadata


def build_report():
    DOC.mkdir(parents=True,exist_ok=True)
    config=read(RUN/'RUN_CONFIG.json');receipt=read(RUN/'SCORE_RECEIPT.json');native=read(RUN/'NATIVE_METRICS.json')
    assert receipt['status']=='COMPLETE' and read(RUN/'BASELINE_REPLAY.json')['passed']
    summary=list(csv.DictReader((RUN/'SUMMARY.csv').open()));sub=list(csv.DictReader((RUN/'SUBGROUPS.csv').open()))
    def row(f,g,q):return next(r for r in summary if r['fold']==f and r['group']==g and r['metric']==q)
    def delta(f,s,g,q):
        v=native[f][str(s)][g];return 100*(v['Bstar'][q]-v['B'][q])
    outcomes=[]
    for f in FOLDS:
        mf=row(f,'all','minFDE');jf=row(f,'all','JFDE')
        both=sum(delta(f,s,'all','minFDE')<0 and delta(f,s,'all','JFDE')<0 for s in SEEDS)
        outcomes.append(dict(fold=f,minFDE_and_JFDE_improved_seeds=both,
            minFDE_gain_mean_cm=-float(mf['delta_mean_cm']),minFDE_gain_seed2035_cm=-delta(f,2035,'all','minFDE'),
            minFDE_improved=int(mf['improved']),minFDE_worsened=int(mf['worsened']),
            JFDE_improved=int(jf['improved']),JFDE_worsened=int(jf['worsened']),
            all_four_improved_seeds=int(mf['four_metrics_improved_seeds'])))
    z1=outcomes[0]
    lines=['# B* 三折五推理种子冻结复核',
        '**三折正式冻结复核，推理种子2035–2039，固定权重，每个种子独立K20。**',
        '**时间顺序披露：本次是在已看到2035上“ZARA1改善、ZARA2和HOTEL恶化”的结果后决定补齐2036–2039。** 2035的原预测、封存、评分不覆盖、不重跑；新增8592个配对，合并后10740个配对，来自同一批2148个测试窗。训练更新为0，没有测试选模、调参、校准或其他后处理。',
        '## 三个问题的回答',
        f"1. ZARA1的minFDE与JFDE在 **{z1['minFDE_and_JFDE_improved_seeds']}/5** 个种子上同时改善。2035的minFDE收益为{z1['minFDE_gain_seed2035_cm']:.6f} cm，五种子平均收益为 **{z1['minFDE_gain_mean_cm']:.6f} cm**。四项同时改善{z1['all_four_improved_seeds']}/5次。"]
    for f in ('zara2','hotel'):
        r=next(x for x in outcomes if x['fold']==f)
        counts={q:sum(delta(f,s,'all',q)>0 for s in NEW_SEEDS) for q in METRICS}
        lines.append(f"2. {f.upper()}：五种子中minFDE恶化{r['minFDE_worsened']}/5、JFDE恶化{r['JFDE_worsened']}/5；仅新增四种子中，JADE/JFDE/minADE/minFDE恶化次数分别为{counts['JADE']}/{counts['JFDE']}/{counts['minADE']}/{counts['minFDE']}（每项分母4）。四项同时改善{r['all_four_improved_seeds']}/5次。")
    dense=row('zara2','N_ge_9','JFDE')
    lines.append(f"ZARA2拥挤组JFDE：改善{dense['improved']}、持平{dense['tied']}、恶化{dense['worsened']}次；五种子配对差{float(dense['delta_mean_cm']):+.6f} ± {float(dense['delta_std_cm']):.6f} cm。")
    avg={q:row('equal_fold_mean','all',q) for q in METRICS}
    lines.append('3. 三折等权后，JADE/JFDE/minADE/minFDE的五种子平均配对差分别为'+
        '/'.join(f"{float(avg[q]['delta_mean_cm']):+.6f}" for q in METRICS)+' cm。'+
        ('四项平均值同时改善。' if all(float(avg[q]['delta_mean_cm'])<0 for q in METRICS) else '未同时改善边缘和联合指标。'))
    if z1['minFDE_and_JFDE_improved_seeds']>=3 and all(next(x for x in outcomes if x['fold']==f)['minFDE_worsened']>=3 and next(x for x in outcomes if x['fold']==f)['JFDE_worsened']>=3 for f in ('zara2','hotel')):
        lines.append('**建议保留冻结B作为跨折主模型，结束这一版B*的迭代。** ZARA1的收益不能代表跨折可靠收益；本轮不追加训练、匹配代价、后处理或种子搜索。')
    else:
        lines.append('方向未满足一致的跨折收益；具体混合方向及次数全部列于主表，不据本轮结果换权重或追加搜索。')
    lines+=['## 五种子主表',
        'B、B*为五种子均值（米）；Δ=100×(B*−B)，报告五个配对差的均值±样本标准差（厘米，ddof=1）。改善/持平/恶化按未舍入值<0/=0/>0判定。“四项同降”为同一seed四项均严格下降的次数。',
        '| 折 | 分组 | 窗/人次（每seed） | 指标 | B均值(m) | B*均值(m) | 配对Δ均值±SD(cm) | 改善/持平/恶化 | 四项同降 |',
        '|---|---|---:|---|---:|---:|---:|---:|---:|']
    for r in summary:
        label=r['fold'].upper() if r['fold']!='equal_fold_mean' else f"折等权({r['fold_count']}折)"
        if r['B_mean_m']:
            values=f"{float(r['B_mean_m']):.9f} | {float(r['Bstar_mean_m']):.9f} | {float(r['delta_mean_cm']):+.6f} ± {float(r['delta_std_cm']):.6f} | {r['improved']}/{r['tied']}/{r['worsened']} | {r['four_metrics_improved_seeds']}/5"
        else:values='N/A | N/A | N/A | N/A | N/A'
        lines.append(f"| {label} | {r['group']} | {r['windows']}/{r['agents']} | {r['metric']} | {values} |")
    lines+=['全体折均值先在每个seed内对三折等权平均，再对五个seed计算均值和样本标准差。密集组只含ZARA1/ZARA2两折；HOTEL没有N≥9窗口，记N/A，不补零。K20分别计分，不合并成K100。五种子的波动仅描述固定权重的推理随机性，不代表跨训练种子、跨数据集的统计不确定性。',
        '## 人数分组（五种子配对）',
        '| 折 | 分组 | 窗/人次（每seed） | 指标 | Δ均值±SD(cm) | 改善/持平/恶化 |',
        '|---|---|---:|---|---:|---:|']
    for r in sub:
        if r['metric'] not in ('JFDE','minFDE'):continue
        values=(f"{float(r['delta_mean_cm']):+.6f} ± {float(r['delta_std_cm']):.6f} | {r['improved']}/{r['tied']}/{r['worsened']}" if r['delta_mean_cm'] else 'N/A | N/A')
        lines.append(f"| {r['fold'].upper()} | {r['group']} | {r['windows']}/{r['agents']} | {r['metric']} | {values} |")
    replay=read(RUN/'BASELINE_REPLAY.json');audit=read(RUN/'RANDOM_INPUT_AUDIT.json');wall=read(RUN/'WALL_TIME.json')
    lines+=['## 冻结身份、随机流和封存',
        '冻结B轮次：ZARA1=6、ZARA2=35、HOTEL=19；完整双分配头轮次分别21/16/15，ablation=False。权重、teacher_identity、CPU头/CUDA教师FP32、优化开关与2035测试完全相同。全部模块eval且冻结。详见WEIGHT_IDENTITIES.json与RUN_CONFIG.json。',
        f"在{audit['windows']}个固定观测窗口记录每个新增seed实际进入推理的个人、共享边和桥噪声指纹。每个非空噪声流在四个seed间有四种不同指纹；用同seed重建输入与实际噪声逐元素完全一致。空边集合不要求变化。检查不新增DDIM调用，桥键保留原始联合样本编号。支持断点时用同seed再次核验噪声指纹。",
        f"新增8592个配对全部校验并封存于{receipt['global_seal_unix']}；完整封存复核于{receipt['global_verification_unix']}；首个新真值读取在{receipt['first_truth_read_unix']}。每窗未来只读取一次用于四个新增seed的独立计分，共2148次；2035本轮真值读取与DDIM调用均为0。",
        f"100项历史B对照核验全部通过（五seed、三折全体和有密集组的两折、四指标），最大绝对误差{replay['maximum_absolute_difference_m']:.12g} m；容差固定atol=rtol=1e−6。各seed按自身历史PER_SEED.csv行核对，不使用五种子均值。",
        '## 执行与归档',
        '命令：`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m residual_frozen_test.run --push-on-complete`。worker显式传入`--worker <fold> --seed <2036..2039>`。逐次真实命令/退出码见COMMANDS.json。单辅助GPU进程、40%显存上限、6GiB启动余量，原UNIV/ETH控制器和暂停队列不变。',
        f"本轮从预测调度至评分完成的墙钟{wall['wall_seconds']:.3f}s（{wall['wall_seconds']/60:.2f}分钟，包含重试/恢复等待，不含首次接口检查和Git推送）；评分{receipt['scoring_wall_seconds']:.3f}s。原2035耗时不并入本次新增推理耗时。",
        '权重、原始观测、人员ID和完整轨迹留在训练主机。公开结果为匿名窗口、计数和指标，路径与测试成员索引脱敏。原2035归档保留不变。',
        '[逐种子CSV](PER_SEED.csv) · [五种子配对汇总](SUMMARY.csv) · [人数分组](SUBGROUPS.csv) · [匿名逐窗](PER_WINDOW.csv) · [随机输入核验](RANDOM_INPUT_AUDIT.json) · [2035复用收据](REUSE_2035_RECEIPT.json) · [封存清单](PREDICTIONS_FROZEN.json) · [B核对](BASELINE_REPLAY.json) · [评分收据](SCORE_RECEIPT.json)']
    (DOC/'README.md').write_text('\n\n'.join(lines).replace('|\n\n|','|\n|')+'\n')
    for name in ('SUMMARY.csv','PER_SEED.csv','SUBGROUPS.csv','PER_WINDOW.csv'):shutil.copyfile(RUN/name,DOC/name)
    csvwrite(DOC/'CONCLUSIONS.csv',outcomes)
    for name in ('RUN_CONFIG.json','SCORE_RECEIPT.json','PRE_SCORE_VERIFICATION.json','BASELINE_REPLAY.json','NATIVE_METRICS.json',
                 'COMMANDS.json','WALL_TIME.json','REUSE_2035_RECEIPT.json','RANDOM_INPUT_AUDIT.json'):
        write(DOC/name,public_metadata(read(RUN/name)))
    seal=read(RUN/'PREDICTIONS_FROZEN.json');seal['rows']=[{k:v for k,v in r.items() if k!='index'} for r in seal['rows']]
    seal['host_full_manifest_sha256']=digest(RUN/'PREDICTIONS_FROZEN.json');write(DOC/'PREDICTIONS_FROZEN.json',seal)
    write(DOC/'WEIGHT_IDENTITIES.json',public_metadata(config['identities']))
    for fold in FOLDS:
        out=DOC/fold;out.mkdir(exist_ok=True);m=read(RUN/fold/'TEST_WINDOWS.json')
        m['rows']=[{k:v for k,v in r.items() if k!='index'} for r in m['rows']];write(out/'TEST_WINDOWS.json',m)
        for seed in NEW_SEEDS:
            write(out/f'EXPORT_TIMING_{seed}.json',read(RUN/fold/f'seed_{seed}/EXPORT_TIMING.json'))
    write(DOC/'ARTIFACT_SHA256.json',{str(p.relative_to(DOC)):digest(p) for p in sorted(DOC.rglob('*')) if p.is_file() and p.name!='ARTIFACT_SHA256.json'})
