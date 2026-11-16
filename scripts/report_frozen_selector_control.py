#!/usr/bin/env python3
"""Render the completed, immutable selector control; never select or sample."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run_frozen_selector_control import atomic_json,read,sha


def main(plan_path):
    started=time.monotonic();plan=read(plan_path);assets=Path(plan['asset_root']);docs=Path(plan['documentation_root'])
    receipt=read(assets/'RUN_RECEIPT.json')
    assert receipt['status']=='COMPLETE' and not receipt['active']
    if (assets/'reports/REPORT_RENDER_RECEIPT.json').exists():raise RuntimeError('report already rendered; no automatic rerun')
    assert receipt['cpu_wall_seconds']<plan['budget_seconds']-30
    data=read(docs/'EIGHT_CELL_METRICS.json');timing=read(docs/'TIMING.json');protection=read(docs/'PROTECTION_BEFORE_AFTER.json')
    frozen=read(docs/'OUTPUTS_FROZEN.json');claim=read(assets/'RUN_CLAIM.json')
    for b in frozen['banks']:assert sha(b['path'])==b['sha256']
    rows=data['per_seed'];summaries=data['summary']
    def stats(d,m,key,group='all'):
        return next(r['metrics'][key] for r in summaries if (r['density'],r['method'],r['group'])==(d,m,group))
    def val(d,m,key,group='all'):return stats(d,m,key,group)['mean']
    def show(d,m,key,group='all',scale=1):
        s=stats(d,m,key,group);return f"{s['mean']*scale:.6f} ± {s['sd']*scale:.6f}"
    methods=['IID20','Greedy20','KMeans20','OneSwap20'];labels={'flow':'Flow','mdn8':'MDN8'}
    contrasts=[('Greedy20','IID20'),('KMeans20','IID20'),('Greedy20','KMeans20'),('OneSwap20','Greedy20')]
    def sign_counts(d,left,right,key):
        pairs=[]
        for seed in range(2035,2040):
            get=lambda m:next(r[key] for r in rows if (r['density'],r['method'],r['group'],r['seed'])==(d,m,'all',seed))
            pairs.append(get(left)-get(right))
        return f'{sum(x<0 for x in pairs)}/{sum(x==0 for x in pairs)}/{sum(x>0 for x in pairs)}'
    delta_gk={d:val(d,'Greedy20','minFDE')-val(d,'KMeans20','minFDE') for d in labels}
    text=['# 冻结候选池选择器对照：零训练 CPU 结果','',
          '**完成状态：COMPLETE。** 两种冻结密度 × 四种选择器 × 五个推理 seed 全部完成；没有训练、优化器、模型加载、新支持采样或 GPU 任务。',
          '本轮完成只表示证据齐全，不表示新选择器通过独立泛化或完整 EBJD 性能验收。','',
          f"原 Greedy 相对 KMeans 的 minFDE 差值（Greedy−KMeans）：Flow {delta_gk['flow']:+.6f} m，MDN8 {delta_gk['mdn8']:+.6f} m。负值有利于 Greedy。",'']
    for d in labels:
        dg=val(d,'OneSwap20','minFDE')-val(d,'Greedy20','minFDE')
        ds=val(d,'OneSwap20','R_support')-val(d,'Greedy20','R_support')
        dp=val(d,'OneSwap20','R_probe')-val(d,'Greedy20','R_probe')
        text.append(f"{labels[d]} 单次交换相对 Greedy：池内风险 {ds:+.6f} m，异种子支持风险 {dp:+.6f} m，真实 minFDE {dg:+.6f} m。三者分别报告，不把经验目标下降等同于真实误差改善。")
    text+=['','## 身份、范围与预先固定的规则','',
           '- 唯一历史证据提交：`95157e97dc4047c1dc80223525164243bda57a70`。',
           f"- 实际执行代码提交：`{receipt['identity']['source_commit']}`；逐文件 SHA 见 SOURCE_IDENTITY.json。",
           '- Flow：best epoch 6 / update 22740，SHA256 `cc71f02c3315bb7e9596600689740c644150b44d62f21b24df2fea6e51ec03a4`。',
           '- MDN8：best epoch 1 / update 3790，SHA256 `b5ebef297c42b7449f1a26a5a163b00a97fa5507a64e214d1613ad31611df8b8`。原运行只完成 4 轮，未追加训练；best 与 last 不混用。',
           '- 复用 ARTIFACTS 登记的 10 份 support bank；M=256、K=20、米制，seeds 2035–2039。每 seed 320 窗口、621 次有效行人出现；132 个单人窗口/132 次出现，188 个多人窗口/489 次出现。',
           '- 两种密度来自同一内部开发集、同一训练 seed 3101。每种密度 3105 条“出现×推理 seed”不等于独立行人，更不等于 3105 次训练重复。',
           '- IID20 复用支持前 20 点；Greedy20 复用原输出，并用冻结的原函数逐值回放。没有重写历史选择器。',
           '- KMeans20：同一 U256 上标准 k-means++/Lloyd，float64、5 次重启、100 次迭代上限、绝对位移阈值 1e-6 m；只按池内平方失真选重启。SHA256 局部 seed，不污染全局 RNG。空簇确定性最远点修复。',
           '- OneSwap20：从原 20 个索引枚举全部合法单点交换，只接受一个全局最佳改进；同成本按移出位置、移入 ID 破局，严格改进阈值 1e-12×(1+|R_old|)。不迭代到收敛，不增加学习参数。',
           '- 原 native_ttst 从热图阈值化概率采样 1000 点，再聚类。这里的 KMeans20 是 U256 常规聚类控制，不是 GDTS TTST 完整复现，也没有调用 GDTS。',
           '- 每份输出仅含选择点、索引、计时和身份。十份输出全部落盘、SHA 冻结后才读取真实 truth；随后所有方法统一 float64、同一 endpoint_metrics 评分。旧 float32 重算单列 HISTORICAL_FLOAT32_REPLAY.json，不混入主表。',
           '- R_probe 使用固定 2035→2036→2037→2038→2039→2035 环，并按 model_kind/scene_id/agent_id/单位关联；只用于事后诊断，不进入选点。','',
           '## 八格主表','',
           '以下为五推理 seed 均值 ± 样本标准差（ddof=1）。误差单位 m；Recall 为 %。', '',
           '| 密度 | 选择器 | minFDE ↓ | Recall .25 ↑ | Recall .5 ↑ | Recall 1.0 ↑ | P50 ↓ | P90 ↓ | P95 ↓ |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for d in labels:
        for m in methods:
            fields=[show(d,m,k,scale=100 if k.startswith('recall') else 1) for k in ['minFDE','recall_0.25','recall_0.5','recall_1.0','P50','P90','P95']]
            text.append('| '+ ' | '.join([labels[d],m,*fields])+' |')
    text+=['','## 池内风险、异种子风险和代价','',
           'R_support、R_probe 是未平方平均最近距离（m），平方失真单位 m²。这里的异种子支持是同一密度在同一窗口的另一批采样，不是新场景泛化。重复率按 (20−精确不同坐标数)/20 计算。','',
           '| 密度 | 选择器 | R_support | R_probe | 平方失真 | 精确重复率 | 平均选择 ms/出现 | P50/P90 ms |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for d in labels:
        for m in methods:
            t=timing['methods'][d+'/'+m]
            text.append(f"| {labels[d]} | {m} | {show(d,m,'R_support')} | {show(d,m,'R_probe')} | {show(d,m,'squared_distortion')} | {val(d,m,'duplicate_rate')*100:.4f}% | {t['mean_s']*1000:.3f} | {t['P50_s']*1000:.3f}/{t['P90_s']*1000:.3f} |")
    text+=['',f"CPU：{timing['cpu']}；NumPy {timing['numpy']}；BLAS/OMP/MKL/NumExpr 均固定 1 线程。KMeans 包含全部 5 次重启，OneSwap20 包含从头 Greedy + 一次交换。没有测密度前向，不能用本表宣称端到端推理提速。"]
    for d in labels:
        t=timing['methods'][d+'/OneSwap_additional']
        text.append(f"{labels[d]} 单交换追加平均/P50/P90：{t['mean_s']*1000:.3f}/{t['P50_s']*1000:.3f}/{t['P90_s']*1000:.3f} ms/出现。")
    text+=['','## 配对差与人数分组','',
           '差值固定为左−右。误差/风险负值改善，Recall 差值以百分点表示、正值改善。逐 seed 完整数值见 PAIRED_DELTAS.csv。','',
           '| 密度 | 对照 | ΔminFDE m | minFDE 负/零/正 seed 数 | ΔRecall .5 百分点 | ΔP95 m | ΔR_probe m |',
           '|---|---|---:|---:|---:|---:|---:|']
    for d in labels:
        for l,r in contrasts:
            diff=lambda k:val(d,l,k)-val(d,r,k)
            text.append(f"| {labels[d]} | {l}−{r} | {diff('minFDE'):+.6f} | {sign_counts(d,l,r,'minFDE')} | {100*diff('recall_0.5'):+.4f} | {diff('P95'):+.6f} | {diff('R_probe'):+.6f} |")
    for group,title in [('N1','单人 N=1'),('Nge2','多人 N≥2')]:
        text+=['',f'### {title}','','| 密度 | 选择器 | minFDE m | Recall .5 % | P95 m |','|---|---|---:|---:|---:|']
        for d in labels:
            for m in methods:text.append(f"| {labels[d]} | {m} | {show(d,m,'minFDE',group)} | {show(d,m,'recall_0.5',group,100)} | {show(d,m,'P95',group)} |")
    part=read(docs/'SUPPORT_SUBSET_PARTITION.json')['rows']
    text+=['','## 支持子集的 0.5 m 分解','','仅对 Greedy/OneSwap 做此分解。KMeans 质心可能不在池内，不能套用子集下界。','','| 密度 | 选择器 | 命中 | 池命中但选择遗漏 | 池未覆盖 |','|---|---|---:|---:|---:|']
    for d in labels:
        for m in ('Greedy20','OneSwap20'):
            rs=[r for r in part if r['density']==d and r['method']==m];total=sum(r['total'] for r in rs)
            values=[sum(r[k] for r in rs) for k in ('hit','selection_miss','pool_miss')]
            assert sum(values)==total
            text.append('| '+ ' | '.join([labels[d],m,*[f'{n}/{total} ({100*n/total:.4f}%)' for n in values]])+' |')
    text+=['','遗漏比例的分母是全部行人出现×seed，不能解释为无需改变密度就必然可恢复的性能空间。','',
           '## 检查、保护与成本','',
           '- 合成检查：40 个小型随机支持与完整枚举单交换对照；K=1/K=M、全同点/重复坐标、非法输入、确定性重启/空簇、唯一支持索引、无输入改写。',
           '- GT 负控：测试对象禁止选择阶段访问 truth，替换 truth 后四种输出逐值不变；NumPy/Python 全局 RNG 不变。实际十份 bank 也严格两阶段处理，冻结收据包含时间和 SHA。',
           '- 构造了质心误差小于所有池内点的反例，明确不向 KMeans 应用支持子集下界。probe 场景重排仍按身份匹配，并拒绝跨密度/跨单位。',
           '- 首次合成检查因一维 pairwise 求和与矩阵沿支持轴顺序求和的末位差异，给数学并列解不同排序而失败；仅修正穷举参考检查的求和顺序。选择算法、配置、阈值均未改，尚未处理真实 bank。失败耗时保留在同一累计预算，完整历史见 RUN_CLAIM.json。',
           f"- {len(protection['files'])} 项历史权重、bank、配置、报告和源码前后 SHA 一致。六层论文工作区 HEAD/status 未改：`{protection['paper_after']['head']}`。",
           f"- 冻结分析累计 CPU 进程墙钟 {receipt['cpu_wall_seconds']:.3f} s（含首次失败、测试、选择、评分、保护核验）；本报告生成成本另计入最终 RUN_RECEIPT。预算 3600 s。optimizer updates=0，GPU tasks=0，GPU seconds=0，加载模型数=0。",'']
    # Evidence-driven interpretation; never tune a selector based on these values.
    if all(v<0 for v in delta_gk.values()):
        conclusion='本次开发集上 Greedy 的平均 minFDE 在两种密度均优于常规 KMeans，说明未平方距离覆盖目标具有值得验证的任务对齐价值；尾部、Recall 和分组取舍仍须按表保留。'
        next_step='冻结本轮表现最稳健的自有 Provider/选择器组合，下一轮仅在新的验证序列和独立训练重复上检验稳定性，再决定是否进入六层 EBJD 联合验证。'
    elif all(v>=0 for v in delta_gk.values()):
        conclusion='常规 KMeans 的平均 minFDE 在两种密度均优于 Greedy，当前专门 Greedy 选择器在真实平均终点误差上的独特优势没有建立；不应继续在同一 dev 调参直到获胜。这不等于 KMeans 在所有指标和成本上支配 Greedy。'
        next_step='以简单 KMeans 作为固定控制，在新的验证序列和独立训练重复上核验自有 Provider 的稳定性，确认后再进入六层 EBJD 联合验证。'
    else:
        conclusion='Greedy 相对 KMeans 的平均 minFDE 优势依赖密度，不能宣称专门选择器普遍优于常规聚类；不能隐藏不利密度或尾部结果。'
        next_step='先固定自有 Provider 和两种标准选择规则，在新的验证序列与独立训练重复上核验“优势是否依赖密度”，不继续在当前 dev 调选择器。'
    text+=['## 结论与唯一下一步','',conclusion]
    for d in labels:
        r25=(val(d,'Greedy20','recall_0.25')-val(d,'KMeans20','recall_0.25'))*100
        p95=val(d,'Greedy20','P95')-val(d,'KMeans20','P95')
        ratio=timing['methods'][d+'/KMeans20']['mean_s']/timing['methods'][d+'/Greedy20']['mean_s']
        text.append(f"{labels[d]} 的明确取舍：Greedy 的 Recall@0.25 比 KMeans 高 {r25:.4f} 个百分点、选择计算约快 {ratio:.2f} 倍；但 P95 高 {p95:.6f} m。Greedy 的未平方池内/异种子风险更低，却没有带来更低的真实平均 minFDE。这说明密度经验风险与真实误差并不自动一致；不能仅用 R_support 证明任务收益。KMeans 的尾部较好、中心分位数及小半径覆盖较差，不能概括为全面更优。")
    for d in labels:
        gt=val(d,'OneSwap20','minFDE')-val(d,'Greedy20','minFDE')
        probe=val(d,'OneSwap20','R_probe')-val(d,'Greedy20','R_probe')
        text.append(f"{labels[d]}：单交换的真实 minFDE {'改善' if gt<0 else '未改善'}（{gt:+.6f} m），异种子支持风险{'改善' if probe<0 else '未改善'}（{probe:+.6f} m）。这只是固定一次标准局部搜索的诊断，不证明增加交换次数会继续改善，也不是原创网络。")
    text+=['','所有数据来自反复观察的内部开发集。没有新增盲测、独立训练种子、完整轨迹预测或六层 EBJD 联合评测；不能宣称独立泛化、替代原 Provider、完整 EBJD 提升或统计显著。未重算 NLL/energy，也未在代表点上冒充密度评分。','',
           '**唯一下一步建议：**'+next_step+' 本轮不自动执行。','',
           '## 复现与资料','',
           '```bash',f"../.conda/rsjg/bin/python -B -u scripts/run_frozen_selector_control.py --plan {Path(plan_path).relative_to(ROOT) if Path(plan_path).is_absolute() else plan_path}",'```','',
           '当前 run 已 COMPLETE，独立 run claim 会拒绝重复执行；未完成任务只允许在同一算法/配置及剩余预算内显式 `--resume`。不能通过重开 run 绕过预算。',
           f'输出 bank、逐出现风险、运行日志与 claim：`{assets}`。Git 仅包含代码、小型表格和收据。OUTPUTS_FROZEN.json 提供十份新输出路径/SHA，INPUT_IDENTITY.json 绑定全部旧输入。','',
           '主表 CSV 包含逐 seed/all/N1/Nge2 全部指标；JSON 含均值与 ddof=1 SD。完成后停止，无训练或自动续跑队列。','']
    report='\n'.join(text)
    (docs/'REPORT_ZH.md').write_text(report);(assets/'reports/REPORT_ZH.md').write_text(report)
    atomic_json(docs/'RUN_CLAIM.json',claim)
    elapsed=time.monotonic()-started
    report_receipt={'report_cpu_seconds':elapsed,'selection_outputs_rehashed':10,'new_selection_calls':0,'new_support_sampling':0,'optimizer_updates':0,'gpu_tasks':0}
    atomic_json(assets/'reports/REPORT_RENDER_RECEIPT.json',report_receipt);atomic_json(docs/'REPORT_RENDER_RECEIPT.json',report_receipt)
    receipt['analysis_cpu_seconds_before_reporting']=receipt['cpu_wall_seconds'];receipt['report_cpu_seconds']=elapsed;receipt['cpu_wall_seconds']+=elapsed
    assert receipt['cpu_wall_seconds']<plan['budget_seconds']
    atomic_json(assets/'RUN_RECEIPT.json',receipt);atomic_json(docs/'RUN_RECEIPT.json',receipt)
    print(json.dumps({'status':'COMPLETE','report':str(docs/'REPORT_ZH.md'),'cpu_wall_seconds':receipt['cpu_wall_seconds']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);main(parser.parse_args().plan)
