"""Read-only speed analysis. No model, optimizer, CUDA work, or process control."""
import csv
import hashlib
import io
import json
import math
import statistics as st
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

SOURCE = Path(__file__).resolve().parents[2]
ASSETS = SOURCE.parent / 'RSJG_EBJD_assets/experiments'
RUN = ASSETS / 'zara2_bcep_v3_stage2_20261120'
AUDIT = ASSETS / 'zara2_bcep_v3_stage2_readiness_20261120'
DOC = SOURCE / 'docs/research/bcep_v3/zara2_trial_20261120'
BASE = '2e96b6f35554ceceb5a04fc08b6259dac75e486b'
evidence = {}


def read(path, label):
    raw = path.read_bytes()
    evidence[label] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    return json.loads(raw)


def lines(path, label):
    if not path.exists():
        return []
    raw = path.read_bytes()
    complete = raw if raw.endswith(b'\n') else raw.rsplit(b'\n', 1)[0] + b'\n'
    evidence[label] = dict(complete_prefix_sha256=hashlib.sha256(complete).hexdigest(),
                           complete_prefix_bytes=len(complete), read_bytes=len(raw))
    return [json.loads(x) for x in complete.splitlines() if x.strip()]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def csv_write(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def quantile(values, q):
    v = sorted(values); pos = (len(v) - 1) * q; lo = math.floor(pos); hi = math.ceil(pos)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def describe(values):
    return dict(n=len(values), mean=st.mean(values), median=st.median(values),
                p10=quantile(values, .1), p90=quantile(values, .9), p95=quantile(values, .95),
                minimum=min(values), maximum=max(values), population_stdev=st.pstdev(values))


def grouping(counts, config):
    groups = []
    for start in range(0, len(counts), config['batch_scenes']):
        micro = counts[start:start + config['batch_scenes']]; a = 0
        while a < len(micro):
            b = a + 1
            while b < len(micro) and max(micro[a:b+1]) * (b + 1 - a) <= config['group_max_slots']:
                b += 1
            groups.append((b-a, max(micro[a:b]))); a = b
    return dict(scenes=len(counts), agents=sum(counts), max_agents=max(counts),
                dense_scenes=sum(n >= 9 for n in counts), physical_groups=len(groups),
                padded_agent_slots=sum(b*n for b,n in groups),
                attention_pair_slots=sum(b*n*n for b,n in groups))


def main():
    # Freeze the growing log prefix once. Subsequent computation uses that prefix.
    started = datetime.now(ZoneInfo('Asia/Shanghai'))
    status = read(RUN/'STATUS.json', 'status')
    launch = read(RUN/'LAUNCH.json', 'launch')
    supervisor = read(RUN/'SUPERVISOR_STATUS.json', 'supervisor')
    frozen = read(RUN/'FROZEN_CONFIG.json', 'frozen_config')
    progress = read(RUN/'joint/PROGRESS.json', 'native_progress')
    telemetry = lines(RUN/'MODULE_UPDATES.jsonl', 'module_updates')
    samples = lines(RUN/'RESOURCE_SAMPLES.jsonl', 'resource_samples')
    history = read(RUN/'joint/EPOCH_METRICS.json', 'epoch_metrics') if (RUN/'joint/EPOCH_METRICS.json').exists() else []
    captured = datetime.now(ZoneInfo('Asia/Shanghai'))
    assert launch['attempt'] == 1, 'Recovery requires a separate attempt-aware timing analysis.'
    assert [r['update'] for r in telemetry] == list(range(1, len(telemetry)+1))
    assert len(telemetry) >= 21 and all(r['domain'] == 'real' for r in telemetry)
    for name, expected in frozen['implementation_files'].items():
        assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() == expected, name
    # Read only the already established split metadata, never trajectories or test data.
    formal = ASSETS/'bcep_ebjd_v2_integrated'
    manifest = json.loads((formal/'FORMAL_RUN.json').read_text())
    plan = json.loads(Path(manifest['input_plan']).read_text())
    desc = plan['folds']['zara2']['split']
    split = read(Path(desc['path']), 'official_split_metadata')
    assert evidence['official_split_metadata']['sha256'] == desc['sha256']
    train = split['train_indices']; assert len(train) == 2681
    # A separate CPU generator reconstructs only epoch 1's order; no global/train RNG is touched.
    import torch
    torch.set_num_threads(1)
    order = [train[i] for i in torch.randperm(len(train), generator=torch.Generator().manual_seed(3201)).tolist()]
    config = frozen['config']; rows = []
    for previous, current in zip(telemetry, telemetry[1:]):
        u = current['update']; epoch = (u-1)//168 + 1
        if epoch != (previous['update']-1)//168 + 1:
            continue  # Inter-epoch intervals contain validation and checkpoint I/O.
        delta = current['unix'] - previous['unix']; assert delta > 0
        if epoch != 1:
            continue  # Later orders also depend on augmentation RNG consumption; do not guess them.
        ids = order[(u-1)*16:u*16]
        counts = [len(split['members'][i]['agent_ids']) for i in ids]
        rows.append(dict(epoch=epoch, update=u, seconds=delta, **grouping(counts, config)))
    stats = describe([r['seconds'] for r in rows])
    full = [r for r in rows if r['scenes'] == 16]
    full_stats = describe([r['seconds'] for r in full])
    blocks = []
    for i in range(0, len(full), 20):
        block = full[i:i+20]
        blocks.append(dict(first_update=block[0]['update'], last_update=block[-1]['update'],
                           observations=len(block), mean_seconds=st.mean(r['seconds'] for r in block),
                           mean_agents=st.mean(r['agents'] for r in block),
                           mean_physical_groups=st.mean(r['physical_groups'] for r in block)))
    by_groups = []
    for n in sorted({r['physical_groups'] for r in full}):
        rr = [r for r in full if r['physical_groups'] == n]
        by_groups.append(dict(physical_groups=n, observations=len(rr), mean_seconds=st.mean(r['seconds'] for r in rr),
                              median_seconds=st.median(r['seconds'] for r in rr), mean_agents=st.mean(r['agents'] for r in rr)))
    correlations = {key: st.correlation([r[key] for r in full], [r['seconds'] for r in full])
                    for key in ['agents','max_agents','physical_groups','padded_agent_slots','attention_pair_slots']}
    # Historical timings are context, not a contemporaneous controlled speed benchmark.
    old = read(formal/'zara2/joint/EPOCH_METRICS.json', 'old_v2_coordinate_epoch_metrics')
    old_steady = old[1:]
    old_summary = dict(epochs=len(old), first_epoch_train_seconds=old[0]['train_seconds'],
                       epoch2_onward_train=describe([r['train_seconds'] for r in old_steady]),
                       epoch2_onward_validation=describe([r['validation_seconds'] for r in old_steady]))
    dev_prior = read(AUDIT/'VALIDATION_PROGRESS.json', 'v3_readiness_validation_timing')
    assert dev_prior['epoch'] == 0 and dev_prior['completed_windows'] == 590
    estimate_train = full_stats['mean'] * 168
    estimate_cycle = estimate_train + dev_prior['seconds']
    recent = st.mean(r['seconds'] for r in full[-20:])
    first = st.mean(r['seconds'] for r in full[:20])
    gpu_samples = [r for r in samples if r.get('pid') == launch['pid'] and r.get('gpu_process_mib') is not None
                   and telemetry[0]['unix'] <= r['unix'] <= telemetry[-1]['unix']]
    assert gpu_samples
    resource = dict(samples=len(gpu_samples), gpu_mib=describe([r['gpu_process_mib'] for r in gpu_samples]),
                    rss_peak_mib=max(r.get('rss_kib', 0) for r in gpu_samples)/1024,
                    interval_seconds=5, allocator_peak=False, utilization_average_available=False)
    hardware = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader,nounits'], text=True).strip()
    module_steps = {name: dict(nonzero_gradient_steps=sum(r['modules'][name]['gradient_l2'] > 0 for r in telemetry),
                              nonzero_parameter_update_steps=sum(r['modules'][name]['parameter_delta_l2'] > 0 for r in telemetry))
                    for name in telemetry[0]['modules']}
    current_epoch = (telemetry[-1]['update']-1)//168 + 1
    summary = dict(snapshot_started=started.isoformat(), snapshot_captured=captured.isoformat(),
        latest_logged_update_time=datetime.fromtimestamp(telemetry[-1]['unix'], ZoneInfo('Asia/Shanghai')).isoformat(),
        runtime_source_commit=BASE, native_status=status['status'], supervisor_status=supervisor['status'],
        completed_epochs=len(history), latest_recorded_optimizer_update=telemetry[-1]['update'],
        current_epoch=current_epoch, logged_real_steps=len(telemetry), analysis_optimizer_updates=0,
        native_status_update_field=status.get('optimizer_updates'), native_progress_update=progress['update'],
        startup_to_first_post_hook_seconds=telemetry[0]['unix']-launch['started_unix'],
        post_hook_interval_seconds=stats, full16_post_hook_interval_seconds=full_stats,
        full16_scenes_per_second=16/full_stats['mean'], first20_mean_seconds=first, recent20_mean_seconds=recent,
        recent20_vs_first20_percent=(recent/first-1)*100, workload_pearson_r=correlations,
        projected_training_seconds_per_epoch=estimate_train, prior_readiness_dev_seconds=dev_prior['seconds'],
        projected_train_plus_dev_seconds=estimate_cycle, projected_total_hours_30_epochs=30*estimate_cycle/3600,
        projected_total_hours_100_epochs=100*estimate_cycle/3600,
        estimated_train_slowdown_vs_old_epoch2_onward_median_percent=(estimate_train/old_summary['epoch2_onward_train']['median']-1)*100,
        old_coordinate_timing=old_summary, resources=resource, hardware_query=hardware,
        module_real_updates=module_steps, implementation_unchanged=True,
        limitations=['Epoch-1 partial post-hook intervals are wall time, not optimizer-kernel timings.',
                     'Inter-epoch intervals excluded; startup and first step excluded from steady intervals.',
                     'Projected cycle uses a prior zero-update v3 readiness validation, not this run validation.',
                     'No profiler: no isolated CSEE/HCRA/GRG, Hungarian, data-loading or telemetry timing.',
                     'No contemporaneous v2 speed control; workload correlations do not establish causality.',
                     'Five-second process GPU memory samples can miss peaks; no allocator peak or average utilization.'])
    out = DOC/('speed_'+started.strftime('%Y%m%d_%H%M%S')); out.mkdir()
    write(out/'SUMMARY.json', summary); write(out/'INPUT_SHA256.json', evidence)
    csv_write(out/'UPDATE_TIMINGS.csv', rows); csv_write(out/'TWENTY_UPDATE_BLOCKS.csv', blocks)
    csv_write(out/'PHYSICAL_GROUP_TIMINGS.csv', by_groups)
    csv_write(out/'OLD_V2_EPOCH_TIMINGS.csv', [{k:r[k] for k in ['epoch','update','train_seconds','validation_seconds']} for r in old])
    csv_write(out/'RESOURCE_SAMPLES.csv', [dict(unix=r['unix'], gpu_process_mib=r['gpu_process_mib'], rss_kib=r.get('rss_kib')) for r in gpu_samples])
    module_table = '\n'.join(f"| {k} | {v['nonzero_gradient_steps']} | {v['nonzero_parameter_update_steps']} |" for k,v in module_steps.items())
    group_table = '\n'.join(f"| {r['physical_groups']} | {r['observations']} | {r['mean_seconds']:.3f} | {r['mean_agents']:.1f} |" for r in by_groups)
    corr_table = '\n'.join(f'| {k} | {v:+.4f} |' for k,v in correlations.items())
    native_timing = ('本次尚无完整epoch的训练/验证计时。' if not history else
                     '本次已完成epoch的原生计时：'+str([{k:h[k] for k in ['epoch','train_seconds','validation_seconds']} for h in history]))
    report = f'''# BCEP–EBJD v3 当前训练速度分析

快照：**{captured.isoformat()}**。本次训练保持 **{status['status']}**；已记录 **{len(telemetry)} 次真实optimizer更新**，完整epoch **{len(history)}**。本分析只读取已有日志与划分元数据，额外optimizer更新 **0**，没有暂停、重启、修改训练源码或恢复其他队列。

**当前平均 {full_stats['mean']:.3f} 秒/次逻辑更新，约 {16/full_stats['mean']:.3f} 窗/秒。** 按168次更新折算，训练部分约 **{estimate_train/60:.2f} 分钟/轮**。加上此前同一v3入口590窗验证的 **{dev_prior['seconds']/60:.2f}分钟**，规划值约 **{estimate_cycle/60:.2f}分钟/轮**。这个完整轮时长是估算，不是本次已完成轮的实测。{native_timing}

## 当前实测

统计第一轮相邻AdamW post-hook时间戳，排除启动到第1次更新以及跨轮验证/存盘间隔。主统计仅含16窗完整逻辑batch；尾部不足16窗的更新在逐更新CSV保留但不混入主均值。

| 项目 | 数值 |
|---|---:|
| 16窗更新间隔样本数 | {full_stats['n']} |
| 平均 / 中位数 | {full_stats['mean']:.3f} / {full_stats['median']:.3f} s |
| P10 / P90 / P95 | {full_stats['p10']:.3f} / {full_stats['p90']:.3f} / {full_stats['p95']:.3f} s |
| 最短 / 最长 | {full_stats['minimum']:.3f} / {full_stats['maximum']:.3f} s |
| 前20 / 最近20次平均 | {first:.3f} / {recent:.3f} s |
| 最近20相对前20 | {(recent/first-1)*100:+.2f}% |
| 启动到首次post-hook（包含初始化） | {summary['startup_to_first_post_hook_seconds']:.3f} s |

这些间隔包括前一step的遥测尾部、下一组数据准备/增广、前向、反向、裁剪和AdamW等实际流程；不能解释为单独optimizer运算时间。post-hook在计算参数增量前取时间，因此有相邻步骤边界的少量错位；第2次间隔还含首次checkpoint写入。前后20次差异会受窗口组成影响，不能单凭它认定逐渐变慢。

## 窗口人数和分组

以独立CPU Generator(seed3201)重建第一轮固定顺序，从官方split只取训练窗口人数；不读取轨迹、不运行模型。每次逻辑更新16窗，按4窗microbatch及group_max_slots={config['group_max_slots']}复现原分组规则。大于32人的单窗仍以单独组处理。下表将真实更新耗时按实际前向/反向分组数汇总。

| 实际分组数/更新 | 样本数 | 平均秒/更新 | 平均有效行人数/更新 |
|---:|---:|---:|---:|
{group_table}

| 工作量描述 | 与耗时的Pearson r |
|---|---:|
{corr_table}

agents为16窗有效人数总和；max_agents为最大单窗人数；padded_agent_slots为各组B×N之和；attention_pair_slots为各组B×N²之和，后者只是注意力规模的描述量，不是FLOPs实测。人数、padding、分组数互相关，以上是观察关联，不能拆分各因素的因果耗时。详见[逐更新CSV](UPDATE_TIMINGS.csv)、[20次更新分块](TWENTY_UPDATE_BLOCKS.csv)。

## 历史参照与时间预算

旧v2 coordinate共有{len(old)}轮原生记录：第1轮训练 **{old[0]['train_seconds']/60:.2f}分钟**；第2至{len(old)}轮训练中位数 **{old_summary['epoch2_onward_train']['median']/60:.2f}分钟**，验证中位数 **{old_summary['epoch2_onward_validation']['median']/60:.2f}分钟**。当前v3训练折算值相对该历史训练中位数为 **{summary['estimated_train_slowdown_vs_old_epoch2_onward_median_percent']:+.2f}%**。

这不是同期同负载A/B速度实验：v3是首轮前缀，旧记录是完整轮，遥测和系统负载也可能不同。不能把差值全部归因于新增模块，也不能拿此前单个16窗只反向的4.376秒验收值当作正式AdamW训练步基线。

此前v3零更新验收以590窗、K20、20步DDIM完成验证，用时 **{dev_prior['seconds']:.3f}秒**。本次正式验证尚未完成时仅借它做容量规划；模型改变硬排列后，实测可能变化。

按当前训练均值加上述验证时间，从启动到30轮约 **{30*estimate_cycle/3600:.2f}小时**，到100轮约 **{100*estimate_cycle/3600:.2f}小时**，另加完整checkpoint写盘、启动和可能的技术恢复。minimum30并不表示必在30轮停止；原生patience15、max100决定实际停止时点。这些是条件外推，不是置信区间或承诺完成时间。

## 显存与瓶颈判断的边界

硬件只读查询：`{hardware}`（依次为名称、总显存MiB、驱动）。本次PID在训练区间有{len(gpu_samples)}个5秒采样：GPU进程used_memory中位数 **{resource['gpu_mib']['median']:.0f}MiB**，采样最大 **{resource['gpu_mib']['maximum']:.0f}MiB**；RSS采样最大 **{resource['rss_peak_mib']:.1f}MiB**。

GPU进程显存含context/reserved，且采样会漏瞬时峰值；它既不是PyTorch allocated峰值，也不代表平均GPU利用率。当前日志没有CUDA事件、kernel/CPU算子profile、数据加载或模块分段计时，无法判断究竟GPU计算、CPU Hungarian、图像处理、同步或遥测谁占主要比例。

源码可确认：正式更新包含分组前后向、硬匹配、多个标量同步，以及每次更新对新增模块梯度/参数增量的审计；验证包含完整20步DDIM。上述工作确实存在，但各自耗时尚未实测。本轮保留冻结FP32/math SDPA、分组容量、日志和网络，不以速度分析为由调整实验。

## 训练是否实际推进

| 模块 | 非零梯度step | 非零参数增量step |
|---|---:|---:|
{module_table}

`STATUS.json.optimizer_updates={status.get('optimizer_updates')}` 是入口启动时写入的静态值，RUNNING期间不逐步刷新，不能用它推断真实更新为零。本快照`PROGRESS.json`为update{progress['update']}（每10步刷新），`MODULE_UPDATES.jsonl`已记录到update{telemetry[-1]['update']}（每次真实step记录）。`LOCAL_TRAIN.log`主要在完整轮训练和验证结束后打印一行，空日志不表示未训练。

结论：现有证据支持“每次更新约{full_stats['mean']:.2f}秒、训练部分约{estimate_train/60:.1f}分钟/轮”，并能量化不同窗口工作量下的波动；尚不足以给三个新增模块分别分摊速度成本。接下来由已有后台程序产生完整轮计时，不新增benchmark、optimizer更新或第二次实验。

## 复现与证据

运行源码固定 `{BASE}`，逐文件哈希已与FROZEN_CONFIG核对；冻结协议完全保持。只读重生成命令（生成新的时间快照）：

```bash
OMP_NUM_THREADS=1 ../.conda/rsjg/bin/python scripts/v3_trial/speed_snapshot.py
```

[机器可读统计](SUMMARY.json) · [输入日志前缀SHA](INPUT_SHA256.json) · [历史逐轮计时](OLD_V2_EPOCH_TIMINGS.csv) · [显存采样](RESOURCE_SAMPLES.csv)。日志仍在增长，输入SHA对应读取时的完整行前缀；不会要求后来增长的整个日志继续匹配旧SHA。权重与候选库留在训练主机。
'''
    (out/'REPORT_ZH.md').write_text(report)
    (out/'.gitattributes').write_text('*.csv -text whitespace=cr-at-eol\n')
    write(out/'ARTIFACT_SHA256.json', {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()})
    for p in out.iterdir():
        assert b'/media/ee615/' not in p.read_bytes() and b'/home/ee615/' not in p.read_bytes()
    print(json.dumps(dict(output=str(out.relative_to(SOURCE)), summary=summary), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
