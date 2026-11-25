"""Archive anonymous results for the common fixed epoch-eight protocol."""
import json
import shutil

from bcep.contracts import digest
from bcep_joint.common import read, write
from joint_flow_fourfold.report import csvwrite, public_metadata, read_csv
from .run import DOC, FOLDS, RUN

LABEL = {'all': '全体', 'N_ge_9': 'N≥9', 'single': '单人', 'multi': '多人',
         'singleton': '单人', 'multi_agent': '多人', 'N_eq_1': '单人', 'N_ge_2': '多人'}
METRICS = ('minFDE', 'JFDE', 'minADE', 'JADE', 'L', 'G')


def format_m(value):
    return '未评' if value is None else f'{value:.9f}'


def format_cm(value):
    return '未评' if value is None else f'{100 * value:+.6f}'


def build_report():
    DOC.mkdir(parents=True, exist_ok=True)
    config = read(RUN / 'RUN_CONFIG.json')
    wall = read(RUN / 'WALL_TIME.json')['seconds']
    lines = [
        '# HOTEL、UNIV 的 K=20 双目标微调',
        '**原测试集上的单种子探索性开发评估。** 训练梯度只使用各折原训练集。'
        '原测试集仅用于固定第 8 轮的终点门槛和条件轨迹验证，原始数据划分及历史 A/B 结果保留。',
        f'两折总墙钟 {wall / 60:.2f} 分钟，含缓存、训练、评分与等待资源的时间。',
    ]
    summaries, windows = [], []
    for fold in FOLDS:
        folder = RUN / fold
        evaluation = folder / 'evaluation'
        training_path = folder / 'TRAINING_RESULT.json'
        training = read(training_path)
        summary = read(evaluation / 'SUMMARY.json')
        assert read(folder / 'COMPLETE.json')['evaluation_sha256'] == digest(evaluation / 'SUMMARY.json')
        assert digest(training['final_checkpoint']) == training['final_checkpoint_sha256']
        assert summary['identity']['checkpoint_sha256'] == training['final_checkpoint_sha256']
        assert summary['output_sha256'] == {name: digest(evaluation / name)
                                            for name in ('PER_WINDOW.csv', 'SUMMARY.csv')}
        dest = DOC / fold
        dest.mkdir(exist_ok=True)
        identity = config['identities'][fold]
        all_group = summary['groups']['all']
        delta = all_group['delta']
        endpoint_pass = summary['endpoint_gate_passed']
        all_improved = summary['four_metrics_improved']
        if all_improved:
            decision = '达到终点门槛，四项同时改善，保留该折方案供独立数据验证。'
        elif endpoint_pass:
            decision = '达到终点门槛，完整轨迹未满足四项同时改善，保留原 B。'
        else:
            decision = '未达到终点门槛，止于终点评分，保留原 B。'
        checkpoint_hash = training["final_checkpoint_sha256"]
        lines += [
            f'## {fold.upper()}',
            f'**{decision}** 全体 minFDE 变化 {format_cm(delta["minFDE"])} cm，'
            f'JFDE 变化 {format_cm(delta["JFDE"])} cm。',
            f'教师为冻结 B 第 {identity["teacher_epoch"]} 轮。学生为固定第 8 轮，'
            f'SHA256 为 `{checkpoint_hash}`。训练窗数 {identity["train_windows"]}。',
            '| 分组 | 窗数／人次 | 指标 | 原 B（m） | 第8轮（m） | 新−B（cm） |',
            '|---|---:|---|---:|---:|---:|',
        ]
        for group, row in summary['groups'].items():
            for metric in METRICS:
                original = row['B'][metric]
                updated = row['new'][metric]
                change = row['delta'][metric]
                lines.append(
                    f'| {LABEL.get(group, group)} | {row["scene_count"]}／{row["agent_count"]} | {metric} | '
                    f'{format_m(original)} | {format_m(updated)} | {format_cm(change)} |'
                )
                summaries.append(dict(fold=fold, group=group, windows=row['scene_count'],
                    agents=row['agent_count'], metric=metric, B_m=original, student_m=updated,
                    delta_cm=None if change is None else 100 * change,
                    endpoint_gate_passed=endpoint_pass, four_metrics_improved=all_improved))
        lines.append(f'[{fold.upper()} 逐窗 CSV]({fold}/PER_WINDOW.csv) · '
                     f'[{fold.upper()} 评分收据]({fold}/ENDPOINT_SCORE_RECEIPT.json) · '
                     f'[{fold.upper()} 训练记录]({fold}/TRAINING_RESULT.json)')
        for name in ('PER_WINDOW.csv', 'SUMMARY.csv'):
            shutil.copyfile(evaluation / name, dest / name)
        windows.extend(dict(row, fold=fold) for row in read_csv(evaluation / 'PER_WINDOW.csv'))
        for name in ('SUMMARY.json', 'ENDPOINT_GATE.json', 'EVALUATION_CONFIG.json'):
            write(dest / name, public_metadata(read(evaluation / name)))
        for phase, receipt_name in (('endpoints', 'ENDPOINT'), ('trajectories', 'TRAJECTORY')):
            receipt_path = evaluation / phase / 'SCORE_RECEIPT.json'
            if receipt_path.exists():
                receipt = read(receipt_path)
                assert receipt['seal_unix'] <= receipt['completed_unix'] <= receipt['first_truth_read_unix']
                write(dest / f'{receipt_name}_SCORE_RECEIPT.json', public_metadata(receipt))
        write(dest / 'TRAINING_RESULT.json', public_metadata(training))
        for name in ('PREFLIGHT.json', 'EPOCH_METRICS.json', 'TOP5.json', 'TRAIN_IDENTITY.json',
                     'TEACHER_REFERENCES.json'):
            for path in folder.rglob(name):
                relative = path.relative_to(folder)
                if 'cache' in '/'.join(relative.parts).lower():
                    continue
                write(dest / name, public_metadata(read(path)))
                break
        for name in ('EPOCH_METRICS.csv', 'EPOCHS.csv', 'TRAIN_HISTORY.csv', 'TRAIN_LOG.csv'):
            for path in folder.rglob(name):
                shutil.copyfile(path, dest / name)
                break
    lines.extend([
        '## 固定配置与复现',
        '同一固定训练种子、AdamW 分组和 16 窗逻辑批次。依赖网络学习率 '
        '1e−4，原修正头最后一层学习率 2e−5，共训练 8 轮；'
        '每个逻辑批次按总有效人次归约边缘项，按总窗口数归约联合项、退化惩罚和 ΔNLL。'
        '教师与学生共用原始个人及共享边噪声，各自按原生 medoid 选择 K=20。',
        '评价使用 seed=2035 的历史 B 全精度结果。'
        '全体 ΔminFDE≤−0.020 m 且 ΔJFDE<0 才解码完整轨迹。'
        '正式 minFDE 按人次加权。L 是各窗逐人最佳终点距离的均值再按窗等权，'
        'G=JFDE−L，按窗等权。两折分别判定。',
        '所有已完成轮次保留可恢复权重。第 8 轮固定用于评价，'
        '中间轮次未做探索集验证，也没有用于排名的验证分数。',
        '`CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 '
        'OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python -m k20_dual_finetune.run`',
        '[固定配置和身份](RUN_CONFIG.json) · [汇总 CSV](SUMMARY.csv) · [匿名逐窗 CSV](PER_WINDOW.csv)',
    ])
    (DOC / 'README.md').write_text('\n\n'.join(lines).replace('|\n\n|', '|\n|') + '\n', encoding='utf-8')
    csvwrite(DOC / 'SUMMARY.csv', summaries)
    csvwrite(DOC / 'PER_WINDOW.csv', windows)
    for name in ('RUN_CONFIG.json', 'WALL_TIME.json', 'START.json'):
        write(DOC / name, public_metadata(read(RUN / name)))
    for path in DOC.rglob('*'):
        if path.is_file():
            assert path.suffix in ('.md', '.json', '.csv') and b'/media/ee615' not in path.read_bytes(), path
    write(DOC / 'ARTIFACT_SHA256.json', {
        str(path.relative_to(DOC)): digest(path)
        for path in DOC.rglob('*') if path.is_file() and path.name != 'ARTIFACT_SHA256.json'
    })
