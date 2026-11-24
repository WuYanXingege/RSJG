"""Chinese development-only endpoint-selection diagnosis and one next step."""
import csv
import shutil
from .common import RUN,DOC,BASE,SOURCE,read,write,digest,canonical_hash,csvwrite
from joint_flow_fourfold.report import public_metadata

LABEL={'all':'全体','all_low_speed':'全员低速','mixed':'动静混合','all_moving':'全员运动',
       'mixed_N2_8':'2–8人混合','all_moving_N2_8':'2–8人全员运动'}
QLABEL={'minFDE':'正式逐人 minFDE','L':'L（个人覆盖，窗均）','G':'G（联合代价，窗均）','J':'J / JFDE（窗均）'}


def build_report():
    DOC.mkdir(parents=True,exist_ok=True)
    config=read(RUN/'RUN_CONFIG.json');receipt=read(RUN/'SCORE_RECEIPT.json');seal=read(RUN/'PREDICTIONS_FROZEN.json')
    assert receipt['status']=='COMPLETE' and read(RUN/'NATIVE_REPLAY.json')['passed']
    stats=read(RUN/'METRICS.json');table=list(csv.DictReader((RUN/'SUMMARY.csv').open()))
    def r(group,stage,q):return next(x for x in table if (x['group'],x['stage'],x['metric'])==(group,stage,q))
    def delta(group,stage,q):
        x=r(group,stage,q);return float(x['delta_cm']) if x['delta_cm'] else None
    def compare(value):return '更高（更差）' if value>0 else '更低（更好）' if value<0 else '持平'
    moving_pre=delta('all_moving','pre','minFDE');moving_post=delta('all_moving','post','minFDE')
    mixed_pre=delta('mixed','pre','J');mixed_post=delta('mixed','post','J')
    lines=['# HOTEL 冻结 B：开发集终点选择诊断',
        '**正式结项决定：保留 B，结束 B* 桥后残差头路线；B 原有桥前 CandidateHead 完整保留。四折 A/B 主线继续既定训练与冻结测试。**',
        '本轮是首次 HOTEL 开发集终点选择诊断：B 第19轮、seed=2035、M=256、K=20，688窗/5203人次。训练更新0，测试访问0，DDIM调用0。已见 HOTEL 测试结果仅是提出本次假设的背景，没有据此改权重、搜索速度阈值或按折切换 B/B*。',
        '**本报告统一 Δ=原生20−均匀20，正值表示原生误差更高。** 历史 B* 报告的 B*−B 是另一差值；本轮不评估 B*。',
        '## 首页结论']
    if moving_pre is not None and moving_post is not None:
        location=('差异在整场选完、尚未过 CandidateHead 时就已存在。' if moving_pre>0 and moving_post>0 else
                  '原生选择在旧头前没有边缘损失，边缘损失出现在旧头后。' if moving_pre<=0<moving_post else
                  '旧头前后方向不同，见两阶段配对值。' if moving_pre*moving_post<0 else
                  '旧头前后均未显示原生20损失正式边缘覆盖。')
        lines.append(f"① **全员运动窗**：旧头前原生20的正式逐人minFDE比均匀K20精确期望{compare(moving_pre)}，Δ={moving_pre:+.6f} cm；旧头后比16份均匀K20均值{compare(moving_post)}，Δ={moving_post:+.6f} cm。{location}")
        lines.append(f"旧头后正式minFDE配对差在16次抽样中的样本标准差为{float(r('all_moving','post','minFDE')['delta_sd_cm']):.6f} cm，仅描述随机整场ID抽样波动。")
    else:lines.append('① 全员运动组为空，证据不足；不以零代替缺失组。')
    if mixed_pre is not None and mixed_post is not None:
        for group in ('mixed','mixed_N2_8'):
            if not stats[group]['windows']:continue
            for stage,label in [('pre','旧头前（均匀20精确期望）'),('post','旧头后（16份均值）')]:
                dj=delta(group,stage,'J');dl=delta(group,stage,'L');dg=delta(group,stage,'G')
                lines.append(f"② **{LABEL[group]}，{label}**：原生20的真实JFDE{compare(dj)}，ΔJ={dj:+.6f} cm = ΔL {dl:+.6f} cm + ΔG {dg:+.6f} cm。")
        restricted=delta('mixed_N2_8','post','J')
        lines.append('限制在2–8人后，旧头后JFDE方向'+('一致。' if restricted is not None and (restricted>0)==(mixed_post>0) and (restricted<0)==(mixed_post<0) else '不一致或该子组缺失；保留分组结果，不合并解释。'))
    else:lines.append('② 混合组为空，证据不足；不以零代替缺失组。')
    # The user's proposed mechanism requires the tradeoff to be present before the head as well as after it.
    if all(v is not None for v in (moving_pre,moving_post,mixed_pre,mixed_post)) and moving_pre>0 and moving_post>0 and mixed_pre<0 and mixed_post<0:
        next_step='可立项一种兼顾个人覆盖的整场选择准则，并把动静混合窗的真实JFDE列为后续开发集验收项。'
        why='同为K20时，原生整场选择在全员运动窗损失边缘覆盖，同时在混合窗保住联合收益；此权衡在旧头前已经出现，旧头后方向仍然存在。'
        route='one_scene_selector_with_personal_coverage'
    elif all(v is not None for v in (moving_pre,moving_post,mixed_pre,mixed_post)) and ((moving_pre<=0<moving_post) or (mixed_pre<=0<mixed_post)):
        next_step='先定位原有 CandidateHead 对候选集合的依赖。'
        why='至少一个重点组的劣化方向是在旧头后才出现，不能把这部分变化归因于整场选择本身。'
        route='diagnose_candidate_head_set_dependence'
    else:
        next_step='维持 B，不据本轮诊断追加新选择器或训练。'
        why='本轮未同时支持“全员运动窗损失边缘覆盖、混合窗保住联合收益”在旧头前后成立；保留分组差值与抽样离散程度，证据不足以据此立项一个新准则。'
        route='retain_B'
    decision=dict(next_step=next_step,reason=why,route=route,training_updates=0,test_access=0,
                  moving_pre_minFDE_delta_cm=moving_pre,moving_post_minFDE_delta_cm=moving_post,
                  mixed_pre_JFDE_delta_cm=mixed_pre,mixed_post_JFDE_delta_cm=mixed_post)
    lines+=['## 唯一下一步',f'**{next_step}** {why}',
        '原始256与选后20的差距混入了候选预算变化，不能单独归责选择器；均匀随机对照也不能证明Flow是瓶颈。16份对照只描述固定权重、固定原始256下的抽样波动，不是训练种子波动或新测试结论。',
        '## 分组与固定口径',
        '观测速度为前8帧相邻7段路程之和÷2.8秒；v≤0.2 m/s为低速，阈值固定且包含等号。组名“全员运动”表示所有有效行人均高于此阈值；“混合”表示同时包含低速与高于阈值者。',
        '| 分组 | 窗数 | 人次 |','|---|---:|---:|']
    for group,value in stats.items():lines.append(f"| {LABEL[group]} | {value['windows']} | {value['agents']} |")
    lines+=['L=每窗各人最佳终点误差的均值；J=先对同一场景的有效行人取均值、再在K20中取最小；G=J−L。L/G/J按窗口等权。L使用FP64均值汇总原生FP32逐人误差，J沿用原生FP32场景归约；G由J−L计算。正式逐人minFDE单独按有效人次加权，不能把L的窗均值当作正式minFDE。',
        '旧头前均匀20使用无放回抽样的精确期望：排序误差a[r]乘以C(256−r,19)/C(256,20)，r=1…237。对每人误差及整场人均误差分别计算，再令E[G]=E[J]−E[L]。实现已通过小规模穷举、K=1、K=M及原生指标等价核对。',
        '旧头后16份均匀20在读取未来前按稳定哈希固定ID，每份均为所有行人共享20个不同原始ID。每份单独调用同一个冻结CandidateHead，其组内平均嵌入按该份20组重算；每份独立K20计分后求均值，不能合并成K320。',
        '## 等预算主表',
        '绝对值为米；配对差为厘米。旧头后“Δ抽样SD”来自16个独立K20对照的分组配对差，ddof=1；旧头前为精确期望，不虚构抽样标准差。',
        '| 分组 | 阶段 | 指标 | 原生20(m) | 均匀20(m) | Δ(cm) | Δ抽样SD(cm) |',
        '|---|---|---|---:|---:|---:|---:|']
    for x in table:
        if x['native_m']:
            vals=f"{float(x['native_m']):.9f} | {float(x['uniform_m']):.9f} | {float(x['delta_cm']):+.6f} | "+(f"{float(x['delta_sd_cm']):.6f}" if x['delta_sd_cm'] else 'N/A（精确期望）')
        else:vals='N/A | N/A | N/A | N/A'
        lines.append(f"| {LABEL[x['group']]} | {'旧头前' if x['stage']=='pre' else '旧头后'} | {QLABEL[x['metric']]} | {vals} |")
    raw=list(csv.DictReader((RUN/'RAW256.csv').open()))
    lines+=['## 原始256覆盖参照与回放核验',
        '原始256只作为覆盖参照，不属于K20等预算比较。',
        '| 分组 | 原始256正式minFDE(m) |','|---|---:|']
    for x in raw:
        if x['metric']=='minFDE':lines.append(f"| {LABEL[x['group']]} | {float(x['value_m']):.9f} |" if x['value_m'] else f"| {LABEL[x['group']]} | N/A |")
    replay=read(RUN/'NATIVE_REPLAY.json');decomposition=read(RUN/'DECOMPOSITION_CHECK.json')
    lines+=['| 原生核验项 | 归档(m) | 本轮(m) | 误差(m) |','|---|---:|---:|---:|']
    for x in replay['checks']:lines.append(f"| {x['metric']} | {x['reference_m']:.12f} | {x['actual_m']:.12f} | {x['difference_m']:+.12g} |")
    lines.append(f"原生核验全部通过，固定atol=rtol=1e−6。逐窗及分组ΔJ=ΔL+ΔG共核验{decomposition['checked_window_and_group_comparisons']}项，最大绝对误差{decomposition['maximum_absolute_error_m']:.12g} m，小于1e−12 m。")
    wall=read(RUN/'WALL_TIME.json');seconds=wall['wall_seconds']
    lines+=['## 冻结资产、执行与归档',
        f"本次起始提交`{BASE}`，实际B依赖权重为HOTEL第19轮，SHA256=`1806aabf478641021584aca02f4040e8e9c2431fe80c457d5adabba259d073d0`。本折共享上游为density第57轮、原CandidateHead第4轮、coordinate第12轮。完整上游身份及源码哈希见RUN_CONFIG.json。本轮仅使用冻结Flow/编码器/原CandidateHead与依赖网络，没有加载B*修正头、没有训练与DDIM。",
        f"原始256坐标缓存实际复用{seal['raw_reused_windows']}窗、重新生成{seal['raw_generated_windows']}窗。每窗只生成一份原始256，共用于原生选择和全部随机对照；20个ID或覆盖摘要没有被当作原始坐标缓存。原生旧头后终点与归档B轨迹末帧的逐窗最大绝对差为{max(r['native_endpoint_replay_max_abs_m'] for r in seal['rows']):.12g} m；本轮未再次解码，DDIM调用0。",
        f"全部688窗预测先封存：seal={receipt['global_seal_unix']}，完整核验={receipt['verification_unix']}，首个开发真值读取={receipt['first_truth_read_unix']}。三者严格按此顺序。评分端仅访问selection_dev未来，共688次；测试访问0。",
        f"实际墙钟{seconds:.3f}秒（{seconds/60:.2f}分钟；详细范围见WALL_TIME.json），CPU评分{receipt['score_wall_seconds']:.3f}秒。执行命令及资源策略见COMMANDS.json与RUN_CONFIG.json；源代码和权重按本次固定身份运行。",
        '原始候选、观测、完整轨迹和权重保留在训练主机。公开CSV只有匿名窗号、分组计数与误差；公开封存清单移除内部数据索引。',
        '[分组汇总](SUMMARY.csv) · [匿名逐窗](PER_WINDOW.csv) · [16次逐窗抽样](PER_REPETITION.csv) · [16次分组抽样](REPETITION_GROUPS.csv) · [原始256参照](RAW256.csv) · [原生回放](NATIVE_REPLAY.json) · [分解核验](DECOMPOSITION_CHECK.json) · [配置](RUN_CONFIG.json) · [评分收据](SCORE_RECEIPT.json)']
    (DOC/'README.md').write_text('\n\n'.join(lines).replace('|\n\n|','|\n|')+'\n')
    for name in ('SUMMARY.csv','RAW256.csv','PER_WINDOW.csv','PER_REPETITION.csv','REPETITION_GROUPS.csv'):
        shutil.copyfile(RUN/name,DOC/name)
    for name in ('RUN_CONFIG.json','SCORE_RECEIPT.json','PRE_SCORE_VERIFICATION.json','NATIVE_REPLAY.json','DECOMPOSITION_CHECK.json',
                 'METRICS.json','WALL_TIME.json','COMMANDS.json','EXPORT_TIMING.json','FROZEN_MODELS.json','ENDPOINT_REPLAY.json','CACHE_AUDIT.json'):
        if (RUN/name).exists():write(DOC/name,public_metadata(read(RUN/name)))
    seal=read(RUN/'PREDICTIONS_FROZEN.json');seal['rows']=[{k:v for k,v in r.items() if k!='index'} for r in seal['rows']]
    seal['host_full_manifest_sha256']=digest(RUN/'PREDICTIONS_FROZEN.json');write(DOC/'PREDICTIONS_FROZEN.json',public_metadata(seal))
    write(DOC/'DECISION.json',decision)
    write(DOC/'ARTIFACT_SHA256.json',{str(p.relative_to(DOC)):digest(p) for p in sorted(DOC.rglob('*')) if p.is_file() and p.name!='ARTIFACT_SHA256.json'})
    return decision
