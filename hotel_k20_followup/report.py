"""Publish the HOTEL follow-up without including predictions or identifiers."""
from shutil import copyfile

from bcep.contracts import digest
from bcep_joint.common import read, write
from joint_flow_fourfold.report import csvwrite, public_metadata, read_csv
from .run import DOC, METRICS, ORIGINAL_DOC, RUN, SCOPE

LABELS = {'all': '全体', 'single': '单人', 'multi': '多人'}


def build_report():
    DOC.mkdir(parents=True, exist_ok=True)
    result = read(RUN / 'RESULT.json')
    gate = read(RUN / 'CONTINUATION_GATE.json')
    config = read(RUN / 'RUN_CONFIG.json')
    wall = read(RUN / 'WALL_TIME.json')['seconds']
    continued = result['continued']
    assert continued == gate['passed']
    epochs = (8, 16) if continued else (8,)
    views = {}
    windows = []
    for epoch in epochs:
        folder = RUN / f'epoch{epoch:02d}'
        destination = DOC / f'epoch{epoch:02d}'
        destination.mkdir(exist_ok=True)
        summary = read(folder / 'SUMMARY.json')
        views[epoch] = summary['groups']
        assert summary['ddim_calls'] == 445
        for name in ('SUMMARY.csv', 'PER_WINDOW.csv'):
            copyfile(folder / name, destination / name)
        windows.extend(dict(row, epoch=epoch) for row in read_csv(folder / 'PER_WINDOW.csv'))
        for name in ('SUMMARY.json', 'EVALUATION_CONFIG.json', 'ENDPOINT_VERIFICATION.json'):
            if (folder / name).is_file():
                write(destination / name, public_metadata(read(folder / name)))
        for phase in ('endpoints', 'trajectories'):
            for name in ('SCORE_RECEIPT.json', 'PREDICTIONS_FROZEN.json', 'PRE_SCORE_VERIFICATION.json'):
                path = folder / phase / name
                if not path.is_file():
                    continue
                write(destination / (phase.upper() + '_' + name), public_metadata(read(path)))
    changes8 = views[8]['all']['delta']
    lines = [
        '# HOTEL 第 8 轮完整轨迹补评与续训判定',
        f'**{SCOPE}。** 固定 seed=2035、K=20，使用 445 窗及 1197 人次。'
        '原报告中 HOTEL、UNIV 均未达到 minFDE 降低 2 cm 的研发目标，本次追加沿用此历史结论。',
        '第 8 轮复用了原封存的修正终点、候选 ID 和行人顺序。'
        '每窗按原始 ID 绑定桥噪声，运行 eta=0 的 20 步 DDIM。完整轨迹封存并核验后读取未来，'
        '末帧以已有 FP32 容差复核。第 8 轮补评训练更新为 0。',
    ]
    if continued:
        lines.append('**第 8 轮四项均严格低于原 B，已按规则从第 8 轮完整断点续训至第 16 轮。**')
        changes = result['delta_16_minus_8_m']
        changed = '、'.join(f'{name} {100 * value:+.6f} cm' for name, value in changes.items())
        if result['extension_further_improved']:
            lines.append(f'**延长训练进一步改善。** 全体 16−8 为 {changed}。')
        elif all(value >= 0 for value in changes.values()):
            lines.append(f'延长训练未带来进一步改善。全体 16−8 为 {changed}。')
        else:
            lines.append(f'延长训练的收益和代价并存。全体 16−8 为 {changed}。')
        if result['full_metrics_16_lower_than_B']:
            lines.append('第 16 轮全体四项指标均低于同种子 B。')
        else:
            worse = '、'.join(name for name in METRICS if views[16]['all']['delta'][name] >= 0)
            lines.append(f'第 16 轮相对 B 未严格改善的指标为 {worse}。')
    else:
        blockers = '、'.join(f'{name} {100 * changes8[name]:+.6f} cm'
                            for name in result['blocking_metrics'])
        lines.append(f'**第 8 轮未触发续训。** 相对原 B 未严格改善的指标为 {blockers}，因此本次止于第 8 轮补评。')
    lines.append('绝对值以米计，差值以厘米计。JADE 与 JFDE 按窗等权，minADE 与 minFDE 按有效人次加权。门槛用未舍入值判定。')
    header = '| 分组 | 窗数／人次 | 指标 | B（m） | 第8轮（m） | 8−B（cm） |'
    if continued:
        header += ' 第16轮（m） | 16−B（cm） | 16−8（cm） |'
    lines.extend([header, '|---|---:|---|---:|---:|---:|' + ('---:|---:|---:|' if continued else '')])
    summary_rows = []
    for group, group8 in views[8].items():
        group16 = views[16][group] if continued else None
        for name in METRICS:
            original, eight = group8['B'][name], group8['new'][name]
            d8 = eight - original
            row = dict(group=group, windows=group8['scene_count'], agents=group8['agent_count'],
                       metric=name, B_m=original, epoch8_m=eight, epoch8_minus_B_cm=100 * d8)
            line = f'| {LABELS[group]} | {row["windows"]}／{row["agents"]} | {name} | {original:.9f} | {eight:.9f} | {100*d8:+.6f} |'
            if group16:
                assert original == group16['B'][name]
                sixteen = group16['new'][name]
                row.update(epoch16_m=sixteen, epoch16_minus_B_cm=100 * (sixteen-original),
                           epoch16_minus_8_cm=100 * (sixteen-eight))
                line += f' {sixteen:.9f} | {100*(sixteen-original):+.6f} | {100*(sixteen-eight):+.6f} |'
            summary_rows.append(row)
            lines.append(line)
    lines.append(
        f'本次共新增 {result["additional_updates"]} 次训练更新，DDIM 调用 {result["ddim_calls"]} 次，'
        f'实际墙钟 {wall / 60:.2f} 分钟。第 8 轮 checkpoint SHA256 为 `{config["parent_sha256"]}`。'
    )
    if continued:
        training = read(RUN / 'continuation/TRAINING_RESULT.json')
        assert digest(training['final_checkpoint']) == training['final_checkpoint_sha256']
        lines.append(f'第 16 轮 checkpoint SHA256 为 `{training["final_checkpoint_sha256"]}`。'
                     '续训读取原训练缓存，恢复 AdamW、更新计数、随机状态及训练顺序。第 9–16 轮完整断点均保留。')
        write(DOC / 'CONTINUATION_TRAINING_RESULT.json', public_metadata(training))
        for name in ('FIRST_UPDATE.json', 'CACHE_VERIFICATION.json', 'TRAIN_IDENTITY.json'):
            write(DOC / name, public_metadata(read(RUN / 'continuation' / name)))
        for name in ('EPOCH_METRICS.json', 'TOP5.json'):
            path = RUN / 'continuation/checkpoints' / name
            if path.is_file():
                write(DOC / name, public_metadata(read(path)))
    lines.extend([
        '复现命令 `CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 '
        'OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python -m hotel_k20_followup.run`。',
        f'[原双折研究报告](../{ORIGINAL_DOC.name}/README.md) · [本轮分组结果](SUMMARY.csv) · '
        '[匿名逐窗 CSV](PER_WINDOW.csv) · [续训判定](CONTINUATION_GATE.json)',
    ])
    (DOC / 'README.md').write_text('\n\n'.join(lines).replace('|\n\n|', '|\n|') + '\n', encoding='utf-8')
    csvwrite(DOC / 'SUMMARY.csv', summary_rows)
    csvwrite(DOC / 'PER_WINDOW.csv', windows)
    for name in ('RUN_CONFIG.json', 'CONTINUATION_GATE.json', 'RESULT.json', 'START.json', 'WALL_TIME.json'):
        write(DOC / name, public_metadata(read(RUN / name)))
    for path in DOC.rglob('*'):
        if path.is_file():
            assert path.suffix in ('.md', '.csv', '.json') and b'/media/ee615' not in path.read_bytes(), path
    write(DOC / 'ARTIFACT_SHA256.json', {
        str(path.relative_to(DOC)): digest(path)
        for path in DOC.rglob('*') if path.is_file() and path.name != 'ARTIFACT_SHA256.json'
    })
