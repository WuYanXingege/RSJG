"""Archive anonymous frozen outcomes, including any adverse effects."""
import csv
import math
import shutil
from .common import *
from joint_flow_fourfold.report import public_metadata


def direction(delta):
    return '改善' if delta<0 else '恶化' if delta>0 else '持平'


def build_report():
    DOC.mkdir(parents=True,exist_ok=True)
    config=read(RUN/'RUN_CONFIG.json');receipt=read(RUN/'SCORE_RECEIPT.json');native=read(RUN/'NATIVE_METRICS.json')
    assert receipt['status']=='COMPLETE' and read(RUN/'BASELINE_REPLAY.json')['passed']
    rows=list(csv.DictReader((RUN/'SUMMARY.csv').open()));sub=list(csv.DictReader((RUN/'SUBGROUPS.csv').open()))
    outcomes=[]
    for fold in FOLDS:
        b=native[fold]['all']['B'];s=native[fold]['all']['Bstar'];dev=read(DEV_DOC/fold/'RESULT.json')
        dev_gain=100*(dev['baseline']['all']['minFDE']-dev['metrics']['all']['minFDE'])
        test_gain=100*(b['minFDE']-s['minFDE']);percent=100*test_gain/dev_gain
        outcomes.append(dict(fold=fold,dev_minFDE_gain_cm=dev_gain,test_minFDE_gain_cm=test_gain,
            gain_retained_percent=percent,all_four_down=all(s[q]<b[q] for q in METRICS),
            joint_both_nonincreasing=s['JADE']<=b['JADE'] and s['JFDE']<=b['JFDE']))
    lines=['# B* 三折冻结测试：联合与边缘轨迹修正',
           '**结论范围：三折正式冻结测试、单推理种子 2035。**',
           '三折均使用预先锁定的同一轮完整双分配修正头；训练更新为0，没有依据测试结果换权重、调参或添加后处理。B与B*来自同一次真实推理的20个场景，不合并种子。']
    for r in outcomes:
        n=native[r['fold']]['all'];delta={q:100*(n['Bstar'][q]-n['B'][q]) for q in METRICS}
        lines.append(f"- **{r['fold'].upper()}**：minFDE{direction(delta['minFDE'])}{abs(delta['minFDE']):.6f} cm；开发集改善{r['dev_minFDE_gain_cm']:.6f} cm，测试保留率{r['gain_retained_percent']:.2f}%。JADE变化{delta['JADE']:+.6f} cm，JFDE变化{delta['JFDE']:+.6f} cm，minADE变化{delta['minADE']:+.6f} cm。四项同时下降：{'是' if r['all_four_down'] else '否'}。")
    lines+=['保留率=(B测试minFDE−B*测试minFDE)/(B开发minFDE−B*开发minFDE)。负值表示测试中反向恶化，超过100%照实保留；不是同分布误差的统计置信区间。',
            '## 测试主表',
            '绝对值为米，Δ=100×(B*−B)为厘米，负值表示改善。联合指标按窗等权，边缘指标按有效行人人次加权；四项分别取最优样本。',
            '| 折 | 分组 | 窗数/人次 | 指标 | B (m) | B* (m) | Δ (cm) |',
            '|---|---|---:|---|---:|---:|---:|']
    for r in rows:
        label=r['fold'].upper() if r['fold']!='equal_fold_mean' else f"折等权（{r['fold_count']}折）"
        vals=[f"{float(r[k]):.9f}" if r[k] else 'N/A' for k in ('B_m','Bstar_m','delta_cm')]
        lines.append(f"| {label} | {r['group']} | {r['windows']}/{r['agents']} | {r['metric']} | {' | '.join(vals)} |")
    lines+=['全体汇总为三折各指标等权平均；密集组汇总只含ZARA1、ZARA2两折。HOTEL没有N≥9测试窗，记N/A，不补零。只有一个推理种子，不报告跨种子标准差或将开发集门槛当成测试选模规则。',
            '## 人数分组解释',
            '| 折 | 人数组 | 窗数/人次 | ΔJFDE (cm) | ΔminFDE (cm) |',
            '|---|---|---:|---:|---:|']
    dev_groups=[]
    for fold in FOLDS:
        for g in ('N_eq_1','N_2_to_8','N_ge_9'):
            b=native[fold][g]['B'];s=native[fold][g]['Bstar']
            if b:lines.append(f"| {fold.upper()} | {g} | {b['scene_count']}/{b['agent_count']} | {100*(s['JFDE']-b['JFDE']):+.9f} | {100*(s['minFDE']-b['minFDE']):+.9f} |")
            else:lines.append(f'| {fold.upper()} | {g} | 0/0 | N/A | N/A |')
        devrows=list(csv.DictReader((DEV_DOC/fold/'PER_WINDOW.csv').open()))
        for g in ('N_eq_1','N_2_to_8','N_ge_9'):
            selected=[r for r in devrows if g in groups_for(int(r['n']))];n=sum(int(r['n']) for r in selected)
            dev_groups.append(dict(fold=fold,group=g,windows=len(selected),agents=n,
                JFDE_delta_cm=100*math.fsum(float(r['student_JFDE'])-float(r['B_JFDE']) for r in selected)/len(selected) if selected else None,
                minFDE_delta_cm=100*math.fsum(float(r['student_minFDE_sum'])-float(r['B_minFDE_sum']) for r in selected)/n if n else None))
    lines.append('ZARA2逐组与同一选中轮次的开发集比较（仅作解释，不重新选模）：')
    for dev in [r for r in dev_groups if r['fold']=='zara2']:
        b=native['zara2'][dev['group']]['B'];s=native['zara2'][dev['group']]['Bstar']
        df=100*(s['JFDE']-b['JFDE']);dm=100*(s['minFDE']-b['minFDE'])
        lines.append(f"- {dev['group']}：开发集ΔJFDE={dev['JFDE_delta_cm']:+.6f} cm，测试ΔJFDE={df:+.6f} cm（{direction(df)}）；开发集ΔminFDE={dev['minFDE_delta_cm']:+.6f} cm，测试ΔminFDE={dm:+.6f} cm（{direction(dm)}）。")
    z=native['zara2'];low=[z[g] for g in ('N_eq_1','N_2_to_8')];count=sum(v['B']['scene_count'] for v in low)
    low_delta=100*sum(v['Bstar']['sums']['scene_JFDE']-v['B']['sums']['scene_JFDE'] for v in low)/count
    dense_delta=100*(z['N_ge_9']['Bstar']['JFDE']-z['N_ge_9']['B']['JFDE'])
    lines.append(f"ZARA2合并N≤8测试窗的ΔJFDE={low_delta:+.6f} cm，N≥9为{dense_delta:+.6f} cm。因此，低人数JFDE上升而拥挤窗JFDE改善的方向在测试中{'复现' if low_delta>0 and dense_delta<0 else '未同时复现'}。分组结果只是解释，未用于训练或测试选模。")
    replay=read(RUN/'BASELINE_REPLAY.json')
    lines+=['## 身份、封存与评分核验',
        f"三折完整2148窗先生成并逐片核验成员哈希、有效人数、FP32形状、有限性、20个唯一原始编号和文件SHA256，再建立全局封存清单；首个测试真值读取时间为{receipt['first_truth_read_unix']}，晚于全局封存{receipt['global_seal_unix']}和完整核验{receipt['global_verification_unix']}。每窗B与B*共用同一次桥解码，覆盖摘要读取本窗原始256组终点。",
        f"B与历史PER_SEED.csv中2035行对照通过，全体/密集四项指标最大绝对差为{replay['maximum_absolute_difference_m']:.12g} m。固定FP32核验容差为atol=rtol=1e−6，未用五种子均值代替对照。",
        '| 折 | 冻结B轮次 | 头轮次 | 头SHA256 |','|---|---:|---:|---|']
    for fold in FOLDS:
        i=config['identities'][fold];lines.append(f"| {fold.upper()} | {i['teacher_identity']['teacher_epoch']} | {i['head_epoch']} | `{i['head_sha256']}` |")
    wall=read(RUN/'WALL_TIME.json')
    lines+=['全部上游和修正头均eval、requires_grad=False；运行前后state hash一致，无优化器或训练更新。保留头checkpoint中的优化开关：关闭geometry,hungarian,logging,modulation,proposal,schedule；头为CPU FP32，教师为确定性CUDA FP32。原训练入口仍不允许测试访问。',
        '## 运行与交付',
        '执行命令：`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m residual_frozen_test.run --push-on-complete`。使用本次控制器锁及辅助GPU锁，单辅助worker，40%显存上限，启动前至少6 GiB余量；UNIV/ETH原控制器未改动。',
        f"从预测调度开始至评分完成的实际墙钟{wall['wall_seconds']:.3f}s（{wall['wall_seconds']/60:.2f}分钟，不含首次接口检查及Git推送）；评分墙钟{receipt['scoring_wall_seconds']:.3f}s。实际各进程命令、时段见COMMANDS.json。",
        '权重、原始观测、完整轨迹和人员ID保留在训练主机。公开窗口清单与预测封存清单只含匿名窗口号、计数、身份哈希及预测文件哈希；JSON中的主机路径经过脱敏。',
        '[汇总CSV](SUMMARY.csv) · [人数分组](SUBGROUPS.csv) · [匿名逐窗CSV](PER_WINDOW.csv) · [配置](RUN_CONFIG.json) · [封存清单](PREDICTIONS_FROZEN.json) · [评分收据](SCORE_RECEIPT.json) · [B回放核对](BASELINE_REPLAY.json)']
    (DOC/'README.md').write_text('\n\n'.join(lines).replace('|\n\n|','|\n|')+'\n')
    for name in ('SUMMARY.csv','SUBGROUPS.csv','PER_WINDOW.csv'):
        shutil.copyfile(RUN/name,DOC/name)
    csvwrite(DOC/'GAIN_RETENTION.csv',outcomes);csvwrite(DOC/'DEV_SUBGROUPS.csv',dev_groups)
    for name in ('RUN_CONFIG.json','SCORE_RECEIPT.json','PRE_SCORE_VERIFICATION.json','BASELINE_REPLAY.json','NATIVE_METRICS.json','COMMANDS.json','WALL_TIME.json'):
        write(DOC/name,public_metadata(read(RUN/name)))
    seal=read(RUN/'PREDICTIONS_FROZEN.json')
    seal['rows']=[{k:v for k,v in row.items() if k!='index'} for row in seal['rows']]
    seal['host_full_manifest_sha256']=digest(RUN/'PREDICTIONS_FROZEN.json')
    write(DOC/'PREDICTIONS_FROZEN.json',seal)
    write(DOC/'WEIGHT_IDENTITIES.json',public_metadata(config['identities']))
    for fold in FOLDS:
        out=DOC/fold;out.mkdir(exist_ok=True)
        manifest=read(RUN/fold/'TEST_WINDOWS.json')
        manifest['rows']=[{k:v for k,v in r.items() if k!='index'} for r in manifest['rows']]
        write(out/'TEST_WINDOWS.json',manifest)
        write(out/'EXPORT_TIMING.json',read(RUN/fold/'EXPORT_TIMING.json'))
    write(DOC/'ARTIFACT_SHA256.json',{str(p.relative_to(DOC)):digest(p) for p in sorted(DOC.rglob('*')) if p.is_file() and p.name!='ARTIFACT_SHA256.json'})
