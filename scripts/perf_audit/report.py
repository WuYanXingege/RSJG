"""Assemble the completed CPU audit; do not manufacture GPU benchmark results."""
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import xml.etree.ElementTree as ET

SOURCE=Path(__file__).resolve().parents[2]
LOCAL=SOURCE.parent/'RSJG_perf_outputs/bcep_v3_speed_audit_20261120'
OUT=SOURCE/'docs/research/bcep_v3/performance_optimization_20261120'
ANCHOR='03158af4a56e75e1cc169b7e8e096048b54ba0bc'
RUNTIME='2e96b6f35554ceceb5a04fc08b6259dac75e486b'
FROZEN='294054c2cf7605caafe016507bb601e2d85518b70555369ca3690b74138e9e43'


def read(p):return json.loads(p.read_text())
def write(name,x):(OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def location(file,qualified):
    found={}
    def walk(node,prefix=''):
        for child in ast.iter_child_nodes(node):
            if isinstance(child,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
                name=prefix+child.name;found[name]=(child.lineno,child.end_lineno);walk(child,name+'.')
    walk(ast.parse((SOURCE/file).read_text()))
    a,b=found[qualified]
    return f'{file}:{qualified}:L{a}–L{b}'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    receipts=[read(LOCAL/name) for name in ['EXACTNESS.json','secondary/EXACTNESS.json','perstep/EXACTNESS.json','edges/EXACTNESS.json','journal/EXACTNESS.json']]
    updates=sum(r['synthetic_cpu_optimizer_updates'] for r in receipts)
    assert updates==78 and all(r['real_optimizer_updates']==0 and not r['cuda_initialized'] for r in receipts)
    tests=[];test_names=set()
    for name in ['CPU','SECONDARY','PERSTEP','EDGES','JOURNAL']:
        root=ET.parse(LOCAL/(name+'.xml')).getroot();cases=root.findall('.//testcase')
        assert all(c.find('failure') is None and c.find('error') is None and c.find('skipped') is None for c in cases)
        test_names.update((c.attrib.get('classname'),c.attrib['name']) for c in cases)
        tests.append(dict(run=name,passed=len(cases),xml_sha256=sha(LOCAL/(name+'.xml'))))
    protection=read(LOCAL/'PROTECTION.json');assert protection['status']=='PASS'
    forward=read(LOCAL/'FORWARD_EXACTNESS.json');assert forward['status']=='PASS_CPU'
    geometry=read(LOCAL/'GEOMETRY.json');noise=read(LOCAL/'CPU_NOISE.json')
    origin=read(SOURCE.parent/'RSJG_EBJD_assets/experiments/zara2_bcep_v3_stage2_20261120/INITIALIZATION_ORIGIN.json')
    new_tensors=len(origin['new']);new_elements=sum(math.prod(v['shape']) for v in origin['new'].values())
    original_config=read(SOURCE/'docs/research/bcep_v3/stage2_ready_20261120/FROZEN_CONFIG.json')
    baseline=SOURCE.parent/'RSJG_BCEP_v3_perf_baseline_20261120'
    for name,digest in original_config['implementation_files'].items():
        assert sha(SOURCE/name)==sha(baseline/name)==digest,name
    points=[
      ('P0','bcep_v3/formal_telemetry.py','ModuleTelemetry.capture_backward','bcep_v3/perf/telemetry.py','BatchedModuleTelemetry.capture_backward',
       '逐张量 finite 布尔读回与 FP64 平方和 float 读回','每模块设备端归约，三个模块的 finite/平方和一次读回；仍在 clip/step 前拒绝 NaN/Inf',
       '不接入损失图；统计跨张量求和顺序可变化，模型/梯度不得变化','逐step状态、非有限负例、缺失梯度、参数对象身份'),
      ('P0','bcep_v3/formal_telemetry.py','ModuleTelemetry.step_done','bcep_v3/perf/telemetry.py','BatchedModuleTelemetry.step_done',
       '逐张量参数差 FP64 范数与最大值读回；rows 无界增长；每步 JSONL','每模块归约后一次读回；保留每步字段/编号/写盘；内存仅留64行',
       '统计允许独立舍入容差；参数克隆与实际step对应不变；没有 fsync 断电保证','正常恢复、重放编号、完整日志、70行内存界限及半行拒绝'),
      ('P1','bcep_v2/stages.py','joint_callbacks.backward','bcep_v3/perf/stages.py','joint_callbacks.backward',
       '每物理组读取 loss、pair metrics 和 loss components 到 Python','detach 后保留标量，逻辑更新末集中读回，按原次序在 Python 进行同样乘除/累加',
       '损失构造、agent/scene 权重、backward顺序、RNG和optimizer完全不变；非有限诊断在step前拒绝','完整16窗+3窗尾批，逐step前后SHA，恢复和RNG，所有日志字段'),
      ('P2','bcep_v2/stages.py','joint_callbacks.backward','bcep_v3/perf/stages.py','compact_checked',
       'physical.to(cuda) 后再 physical.valid.cpu() 与CPU掩码比较','先验证CPU压紧后掩码相同，再搬运同一 SceneBatch；保留非前缀mask拒绝',
       '没有数值运算或成员重排；counts.tolist本来就在CPU，不记作CUDA同步','坏padding在.to前拒绝、混合人数、顺序和完整状态精确相等'),
      ('P3','ebjd/encoders.py','MotionSceneContextEncoder.forward','bcep_v3/perf/options.py','checkpoint_arm',
       '训练时按map_agent_chunk对U-Net activation checkpoint重算','仅独立性能臂显式切换encoder/denoiser原有开关，并finally恢复',
       'CPU小图2人样本输出/梯度精确；GPU全分辨率57人显存和数值均未验收；默认仍开checkpoint','独立map/bridge/both臂，输出/梯度/RNG与未来allocator peak'),
      ('P4','bcep_v3/geometry.py','reference_paths','scripts/perf_audit/mechanisms.py','simplified',
       'CV baseline + 比例×端点残差','仅诊断脚本实现代数化简；生产geometry、incidence和attention均未改',
       'FP32非逐位等价；57人发生argmin时间跳变，拒绝默认替换','12维逐项差、CPA、世界分数、两次Hungarian、梯度与近并列负对照'),
      ('P5','bcep_joint/inference.py','keyed_noise','scripts/perf_audit/mechanisms.py','noise_audit',
       '590窗4262人×20次CPU Generator初始化','只读CPU缓存可行性实验；键包含scene/agent/world/seed/有效顺序/帧坐标身份；不接入generate',
       '缓存副本逐字节相同，CPU RNG不变；小成本不等于整网显著加速','590窗三次原生/缓存对照、字节hash、缓存8MB成本')]
    change=[]
    for priority,oldfile,oldfn,newfile,newfn,old,principle,risk,evidence in points:
        change.append(dict(priority=priority,original=location(oldfile,oldfn),candidate=location(newfile,newfn),
                           original_work=old,change=principle,numerical_boundary=risk,required_evidence=evidence))
    trace=dict(status='CPU_READY_GPU_PROFILE_PENDING',speedup_percent=None,speedup_confidence_interval=None,
        measured_gpu_benchmark=False,gpu_profiler_attached=False,formal_controller_at_initial_check=1978816,
        initial_check_latest_progress_update=290,gpu_busy_at_initial_check=True,
        normal_fully_populated_gradient_telemetry=dict(new_parameter_tensors=new_tensors,
            original_host_scalar_boundaries=4*new_tensors,candidate_host_transfers=2,
            parameter_clone_float32_bytes=4*new_elements,statistic_reduction_dtype='float64',
            warning='Static normal-path boundary count, not measured latency or a speedup prediction.'),
        diagnostic_tensor_scalar_reads_per_physical_group=11,candidate_logical_update_host_transfers=1,
        unchanged_fail_fast=['native gradient clipping error_if_nonfinite','native finite_tensors post-step batch check','v3 asynchronous input assertions'],
        cpu_collation_counts_tolist_is_cuda_sync=False,
        profile_hypotheses=[
            dict(location=location('ebjd/denoiser.py','GeometryConditionedJointDenoiser.forward'),scope='six FJAB checkpoint recomputation; training only'),
            dict(location=location('ebjd/encoders.py','MotionSceneContextEncoder.forward'),scope='map U-Net checkpoint recomputation; context encoded once for validation'),
            dict(location=location('ebjd/unified.py','hard_assignment_indices'),scope='scores GPU-to-CPU, CPU Hungarian twice, per-agent result copies; unchanged'),
            dict(location=location('bcep_v3/coupling.py','RelationalAssignment.relations'),scope='edge-chunk candidate pair geometry and dense incidence/matmul; unchanged'),
            dict(location=location('bcep_v3/coupling.py','RelationalAssignment.forward'),scope='dense world social attention and relation values; unchanged'),
            dict(location=location('bcep_v3/gate.py','GeometryReliabilityGate.forward'),scope='actual path pair distances and bounded CTF gate; unchanged'),
            dict(location=location('bcep_v2/objective.py','bridge_and_pair_loss'),scope='training pairing K20 but conditional bridge uses P=1 target-conditioned world'),
            dict(location=location('bcep_joint/inference.py','generate'),scope='validation K20, DDIM20; one shared context; no change')],
        changes=change,geometry_status=geometry['status'],noise_cpu_only=True,
        no_claimed_causal_attribution_from_population_correlations=True)
    exact=dict(status='PASS_CPU_GPU_PENDING',tests=tests,unique_test_cases=len(test_names),executed_test_cases=sum(r['passed'] for r in tests),
        synthetic_cpu_optimizer_updates=updates,synthetic_gpu_optimizer_updates=0,real_data_optimizer_updates=0,
        baseline_fixed_commit=ANCHOR,formal_source_commit=RUNTIME,synthetic_pixels=32,
        cpu_receipts=receipts,raw_forward=forward,
        telemetry_statistic_tolerance=dict(relative=1e-12,absolute=1e-15,max_observed_absolute=4.336808689942018e-19),
        model_state_tolerance=0,optimizer_state_tolerance=0,CUDA_RNG_equivalence='PENDING',
        GPU_max57_full256_map_capacity='PENDING',research_geometry_equivalence='NON_BITWISE_OPTIONAL')
    write('TRACE_SUMMARY.json',trace);write('EXACTNESS.json',exact);write('PROTECTION.json',protection)
    write('GEOMETRY.json',geometry);write('CPU_NOISE.json',noise)
    write('AUDIT_CONFIG.json',dict(report_anchor=ANCHOR,runtime_anchor=RUNTIME,formal_frozen_sha256=FROZEN,
        branch='perf/bcep-v3-speed-audit-20261120',baseline_worktree='RSJG_BCEP_v3_perf_baseline_20261120',
        candidate_worktree='RSJG_BCEP_v3_perf_audit_20261120',status='CPU_READY_GPU_PROFILE_PENDING',
        optional_defaults=dict(batched_telemetry=False,defer_diagnostics=False,cpu_mask_check=False,
            map_checkpoint=True,bridge_checkpoint=True,simplified_geometry=False,noise_cache=False),
        real_updates=0,synthetic_cpu_updates=updates,synthetic_gpu_updates=0,
        formal_entry_and_identity_guards_unchanged=True,formal_entry_does_not_import_perf=True))
    (OUT/'CHANGE_MAP.md').write_text('# 改动与保留位置\n\n原始位置相对于固定报告提交；候选均是隔离分支的新文件。现有正式文件未改。\n\n'+
        '\n\n'.join(f"## {r['priority']}\n\n- 原始：`{r['original']}`\n- 候选：`{r['candidate']}`\n- 原计算：{r['original_work']}。\n- 具体改动：{r['change']}。\n- 数值边界：{r['numerical_boundary']}。\n- 证据：{r['required_evidence']}。" for r in change)+
        '\n\n## 保留的科学和安全路径\n\n'+ '\n'.join(f"- `{r['location']}`：{r['scope']}。" for r in trace['profile_hypotheses'])+
        '\n\n物理组不按人数重排；不使用BF16、TF32、torch.compile、新DataLoader进程或GPU并行。CSEE/HCRA/GRG-CTF、旧损失、Provider、候选、硬排列和DDIM语义保持。\n')
    commands='''# 从隔离的 perf 分支工作区执行；每次复查选择新的输出子目录。
# CPU 单测：隐藏全部 GPU；只使用合成数据，包含原生恢复/Top5/early-stop测试。
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BCEP_PERF_OUTPUT=../RSJG_perf_outputs/bcep_v3_speed_audit_20261120/cpu_recheck_manual nice -n 19 ../.conda/rsjg/bin/python -m pytest -q tests/test_bcep_v3_perf.py tests/test_bcep_v3.py tests/test_bcep_v3_formal.py tests/test_v3_perf_admission.py -o addopts=''

# 当前不执行。只有正式控制器已停止且GPU无其他计算进程时才可能通过入口检查。
# 2次预热、每形状5次测量、256像素地图；包括57人及混合16窗。
# 附加 checkpoint 开关和关闭 telemetry 仅是性能臂，不是推荐默认。
RUN_SYNTHETIC_PERF=true OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 ../.conda/rsjg/bin/python scripts/perf_audit/benchmark.py --mode throughput --warmup 2 --repetitions 5 --include-checkpoint-arms --telemetry-off-control --output ../RSJG_perf_outputs/bcep_v3_speed_audit_20261120/gpu_throughput_manual

# 单独高成本profile；不能把profile时长当成吞吐。仍需GPU空闲，仍不会排队等待。
RUN_SYNTHETIC_PERF=true OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 ../.conda/rsjg/bin/python scripts/perf_audit/benchmark.py --mode profile --shape 1x57 --shape mixed16 --warmup 2 --repetitions 5 --output ../RSJG_perf_outputs/bcep_v3_speed_audit_20261120/gpu_profile_manual
'''
    (OUT/'COMMANDS.txt').write_text(commands)
    noise_mean=statistics.median(noise['native_seconds']);cache_mean=statistics.median(noise['cached_clone_seconds'])
    geometry_rows='\n'.join(f"| {r['n_agents']} | {r['path_max_abs']:.9g} | {max(r['relation_feature_max_abs']):.9g} | {r['argmin_time_changed_elements']} | {r['scores_max_abs']:.9g} | {r['gradient_max_abs']:.9g} | {r['initial_permutation_equal']}/{r['final_permutation_equal']} |" for r in geometry['rows'])
    report=f'''# BCEP–EBJD v3 独立性能审计与可选优化

**CPU_READY_GPU_PROFILE_PENDING。GPU加速率 UNMEASURED，没有GPU置信区间，也没有BENCHMARK_PASS。** 正式训练状态未被本任务修改；本审计向原训练注入的真实optimizer更新为 **0**。独立合成CPU optimizer更新 **{updates}**，合成GPU更新 **0**。原授权任务在两次资产收据之间自然从update **{protection['native_logged_update_before']}→{protection['native_logged_update_after']}**，PID仍为1978816、状态RUNNING；这部分正常增长不伪称“计数完全不变”。

当前可以提交的是默认关闭的 **P0遥测批量读回、P1诊断集中读回、P2 CPU身份检查**，作为下一次独立实验的候选。它们须先完成空闲GPU上的逐更新精确性和同输入速度/显存验收，再取得新实验授权及重新冻结，不能直接替换当前任务。**Checkpoint关闭开关只用于基准；几何化简不具备逐位等价性；noise缓存本轮不接入。**

## 固定身份与边界

- 报告/代码分叉锚点 `{ANCHOR}`；正式运行源码 `{RUNTIME}`；正式冻结配置规范化SHA `{FROZEN}`。
- 分支 `perf/bcep-v3-speed-audit-20261120`，候选与未修改baseline分别位于独立worktree。所有原有正式源码文件与冻结manifest、baseline逐文件一致；只新增perf命名空间、审计脚本和测试。原formal入口未导入perf，授权与源码SHA检查未放宽。
- 首次只读检查确认正式进程和GPU占用，故本轮没有GPU基准、attach profiler、等待GPU、自动预约、信号、重启或热替换。数值验收和CPU噪声实验隐藏CUDA并使用单线程/低优先级。
- {protection['static_files']}个静态受保护文件前后大小/SHA相同，包括运行worktree、冻结配置、既有完整epoch、历史checkpoint和bank。活动日志、last/best/Top5及新epoch由原任务正常推进，另存前后收据，不纳入“静态文件完全相同”的伪断言。[保护收据](PROTECTION.json)
- v2五折仍PAUSED_RESUMABLE，candidate_cost仍SESSION_BUDGET_PAUSED，sharedworld仍COMPLETE。未打开真实轨迹或test/heldout作训练/评测；P5仅读取已冻结开发集身份元数据和生成CPU随机噪声。

## 证据能说明什么

固定11:45快照的160个更新间隔平均5.8647998765秒，估算训练16.42分钟/轮、加历史验证约21.21分钟/轮；它们不是此次优化收益的测量。人数、padding和分组数高度相关，r≈0.9707不能给某个模块分配因果耗时。本轮确认了同步位置与计算规模，**尚未测得GPU时间占比或最主要瓶颈**。

P0原路径在全部{new_tensors}个新增参数张量都有梯度时有约{4*new_tensors}次标量/布尔host边界：梯度finite、平方和、参数差平方和、参数差最大值各一组。候选保留相同FP64定义，在GPU归约后每逻辑更新集中读回两次。参数快照仍克隆约{4*new_elements}字节；未靠删除审计、降精度或减少写盘取得假提速。边界计数下降不等于相同比例的时间下降，FP64计算和kernel launch仍存在。

P1每物理组约11个CUDA诊断标量改为一个逻辑更新末集中搬回。保留原Python双精度乘除/累加次序，**原loss构造、agent/scene权重、backward调用顺序、裁剪、AdamW与RNG消耗没有更改**。只缓存detach后的标量，不持有autograd图。非有限诊断在本次step前拒绝；原裁剪和参数finite检查保留。

P2把合法成员/压紧mask检查放在搬运前；没有删除验证或重排成员。`counts=batch.valid.sum(1).tolist()`来自CPU collation，不能算CUDA同步。

其他路径仍需独立profile：两次CPU Hungarian及score/result传输、HCRA的E×20×20×12候选对几何、incidence/dense attention、CSEE候选注意力、GRG路径几何、地图U-Net、六层FJAB及重算。原生`finite_tensors`已经统一归约再检查，不再重复优化。v3输入验证使用CUDA异步assert，保留其安全边界。

**训练桥实际使用P=1的目标条件世界，配对仍为K20；验证才是K20、20步H-DDIM。** 验证共享context只编码一次；activation checkpoint仅在训练且开启梯度时生效。因此不能把每次训练步误算为20世界DDIM，也不能指望关闭checkpoint直接减少验证的20步。

逐位置的原计算、候选原理、浮点风险和证据见[CHANGE_MAP.md](CHANGE_MAP.md)，包括文件、函数与行范围。[TRACE_SUMMARY.json](TRACE_SUMMARY.json)没有填写虚构耗时占比。

## 两类等价性

**P0/P1/P2：CPU训练状态精确。** 共{len(test_names)}项不同测试通过，共执行{sum(t['passed'] for t in tests)}个测试用例（含增强验收复跑）。覆盖1、2、9、57人、padding、空邻居、同位邻居、逐人独立K20候选重排、行人重排、混合场景、完整16窗及3窗尾批。比较优化前后同一输入，而不把有并列的Hungarian说成唯一解。

每个完整逻辑update的optimizer前后模型/buffer、梯度、AdamW、CPU/Python/NumPy RNG、顺序/增广生成器及组成员SHA完全相同；原生连续4步、独立P0、P012与1+3步恢复的完整状态相同。scheduler、cursor/order、history/stale与计数也比较。原minimum30/patience15、相同分数选早epoch、Top5与完整原子恢复状态的原生测试通过。统计字段单独容许rtol1e-12/atol1e-15，最大观察差为 **4.336808689942018e-19**，不拿统计容差掩盖模型差异。

额外零optimizer检查直接捕获初/终Hungarian、世界score、硬端点、桥final/coarse/coarse path/coarse goal和全参数梯度：P012在1/2/9/57人合成样本逐张量相同。所有这些CPU样本使用**32像素合成地图**和原192宽度/六层结构，不能代替正式256像素GPU容量与CUDA RNG验收。[EXACTNESS.json](EXACTNESS.json)

P0日志仍每step写完整字段，内存仅保留最近64行；70行压力检查确认不积累历史Tensor。恢复从checkpoint的update编号继续，保留已经存在的物理重放事件而不改写它们；重放测试序列1/2/3/2的统计对应正确。半行尾记录拒绝继续拼接，需要明确离线修复。close不是fsync，突然断电可能丢末尾记录；本轮没有宣称日志提供断电级持久性。70次及4次日志压力/重放中的参数扰动不属于optimizer.step，已与{updates}次合成optimizer更新分开。

**P3/P4：可选计算路径，不能由CPU局部检查外推GPU无损。** Checkpoint开/关在2人32像素CPU样本输出、梯度及RNG一致；map、bridge、both开关都会恢复，默认仍开启checkpoint。没有allocator peak和57人256像素GPU结果，不允许默认关闭。

几何代数化简在实数域等价，但本轮FP32已经出现非零输出/梯度差，状态标为 **NON_BITWISE_OPTIONAL**，只留在诊断脚本中，正式geometry没有改。

| 人数 | 参考路径最大差 m | 12维关系最大差 | argmin时刻改变元素 | 世界score最大差 | 梯度最大绝对差 | 初配/终配相同 |
|---:|---:|---:|---:|---:|---:|---|
{geometry_rows}

57人有2个候选对的最小距离时刻跳变；时间特征约变化1/12，说明微小坐标舍入可经argmin放大。测试样本的两次Hungarian仍相同，硬端点、noise逆gather和边缘minFDE保持。另构造score仅变化1e-8、满足1e-6 allclose却翻转Hungarian的负对照，证明不能用score接近代替排列一致。逐维CPA/关系误差、相对梯度误差分母下限1e-8及负对照说明见[GEOMETRY.json](GEOMETRY.json)。无profile证据支持改动incidence、分块或关系组装，本轮保留原实现。

## Keyed noise与资源成本

按官方590窗、4262个有效行人、K20、seed2035，三次纯CPU生成中位数 **{noise_mean:.4f}秒**；缓存clone中位数 **{cache_mean:.6f}秒**。缓存Tensor需{noise['tensor_cache_bytes']}字节，另有Python键/字典开销。三次输出逐字节相同且CPU全局RNG不变；缓存键涵盖算法/torch版本、scene、agent、world、seed、有效成员顺序、frame IDs和坐标身份。

这只是同进程CPU组件的探索性三次计时，未测整网加速。约1.9秒与历史287.28秒验证相比只占约0.7%的量级，且两者并非同期隔离A/B，故不把缓存接入生产推理。[CPU_NOISE.json](CPU_NOISE.json)

P0保留全部参数克隆，额外GPU归约标量很小但未实测peak；P1保留detach标量至逻辑步末，16个物理组时最多约176个Tensor标量，仍有Python容器与allocator开销。CPU等价性样本不证明关闭U-Net/FJAB重算后能装入显存。原约5.5GiB的进程采样含reserved/context且会漏峰值，不能作为关闭checkpoint的容量证明。

## 待执行的独立GPU基准

当前 **GPU_PROFILE_PENDING**，没有生成BENCHMARK.csv或GPU trace。已提供可执行入口及通过的准入负例测试；**GPU执行分支本身仍未运行**。正式任务未停止、GPU存在其他计算进程、缺少独立授权环境变量、输出路径不在性能目录或目录已存在时均拒绝；不发信号、不等候、不预约。只接受内置合成数据，没有正式训练/数据入口。

未来手动吞吐命令使用同一初态和RNG序列、FP32/math-SDPA、AdamW及原epoch1学习率，形状含1×1、4×8、1×57和混合16窗，地图256像素，2次预热和每形状5次测量。比较baseline、P0、P01、P012，再单列checkpoint开关和关闭telemetry的开销对照。CPU整状态已单测P0与P012组合；P01中间臂和所有GPU臂的独立基准仍待执行。每次调用合成更新上限300；profile独立运行，不能把重profile时间用于速度结论。

记录墙钟、CUDA跨度、p50/p90/方差/范围、max_memory_allocated及实际合成step ledger；逐step完整状态hash在计时区外核验。CUDA跨度含CPU供给造成的空隙，不等于kernel纯占用时间。profile通过独立scope区分数据/增广、两次Hungarian、几何、CSEE/HCRA、map U-Net、六层FJAB前向/重算、GRG、backward、遥测/日志和AdamW；重叠scope不能直接相加。速度差只能取同环境同输入的100×(Tbase−Topt)/Tbase，5次单次调用也不足以声称跨运行稳定收益。

[COMMANDS.txt](COMMANDS.txt)列出手动命令。本轮没有运行这些GPU命令，也没有安排训练结束后自动执行。

## 后续适用范围

保留旧实现为默认。优先在**下一次获准的独立验证**中检验P0，再加入P1/P2；只有GPU状态逐位相同且收益/显存可接受，才考虑新的冻结训练配置。P3保持诊断开关；P4拒绝视为无损；P5不接入。人数重排、BF16、TF32、compile、多进程DataLoader、GPU并行均未实施。

所有正式权重、候选和既有报告保留原位；本任务到此结束，不自动应用补丁、恢复队列、扩折、换seed、消融或测试集选模。
'''
    (OUT/'REPORT_ZH.md').write_text(report)
    log='\n\n'.join('=== '+name+' ===\n'+(LOCAL/(name+'.log')).read_text() for name in ['CPU','SECONDARY','PERSTEP','EDGES','JOURNAL','FORWARD','MECHANISMS'])
    log=log.replace(str(SOURCE),'<ISOLATED_WORKTREE>').replace(str(SOURCE.parent),'<WORKSPACE>')
    (OUT/'CPU_ACCEPTANCE.txt').write_text(log)
    write('ARTIFACT_SHA256.json',{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='ARTIFACT_SHA256.json'})
    for p in OUT.iterdir():
        if p.is_file():assert b'/media/ee615/' not in p.read_bytes() and b'/home/ee615/' not in p.read_bytes(),p.name
    print(json.dumps(dict(status=trace['status'],unique_tests=len(test_names),test_executions=sum(t['passed'] for t in tests),synthetic_updates=updates,real_updates=0,protected=protection['static_files'])))


if __name__=='__main__':main()
