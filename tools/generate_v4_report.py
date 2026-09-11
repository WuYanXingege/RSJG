#!/usr/bin/env python3
"""Generate a V4 experiment report from structured training/evaluation logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path, required: bool = True):
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    with path.open() as handle:
        return json.load(handle)


def _resolve_validation_diagnostics(run_dir: Path, best_epoch: int) -> Path:
    """Return the diagnostic for the validation seed used for model selection.

    Early V4 runs used ``valid_epoch_NNN.json``. Packed-cache runs include the
    cached seed index in the filename and perform model selection with seed 0.
    Supporting both layouts keeps reports reproducible across those runs.
    """
    diagnostics_dir = run_dir / 'diagnostics'
    legacy_path = diagnostics_dir / f'valid_epoch_{best_epoch:03d}.json'
    if legacy_path.exists():
        return legacy_path

    seed_zero_path = diagnostics_dir / (
        f'valid_epoch_{best_epoch:03d}_seed_00.json')
    if seed_zero_path.exists():
        return seed_zero_path

    candidates = sorted(diagnostics_dir.glob(
        f'valid_epoch_{best_epoch:03d}_seed_*.json'))
    if candidates:
        return candidates[0]

    expected = diagnostics_dir / (
        f'valid_epoch_{best_epoch:03d}[_seed_XX].json')
    raise FileNotFoundError(
        f'No validation diagnostics found for epoch {best_epoch}: {expected}')


def _fmt(value, digits=5):
    if value is None:
        return '未记录'
    if isinstance(value, bool):
        return '是' if value else '否'
    if isinstance(value, (int, float)):
        return f'{value:.{digits}f}'
    return str(value)


def _display_path(value):
    """Render project paths without leaking machine-specific absolute roots."""
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        return str(path)
    for root in (PROJECT_ROOT, PROJECT_ROOT.parent):
        try:
            return str(path.relative_to(root))
        except ValueError:
            continue
    return path.name


def _bucket_table(title, buckets):
    lines = [f'### {title}', '',
             '| 分桶 | count | Raw_JADE | JADE | Raw_JFDE | JFDE | recovery_ratio |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for label, values in buckets.items():
        lines.append(
            f"| {label} | {values.get('count', 0)} | "
            f"{_fmt(values.get('Raw_JADE'))} | {_fmt(values.get('JADE'))} | "
            f"{_fmt(values.get('Raw_JFDE'))} | {_fmt(values.get('JFDE'))} | "
            f"{_fmt(values.get('recovery_ratio'))} |")
    if not buckets:
        lines.append('| 暂无有效样本 | 0 | - | - | - | - | - |')
    return lines


def generate(run_dir: Path, output: Path):
    summary = _read_json(run_dir / 'training_summary.json')
    protocol = _read_json(run_dir / 'evaluation_protocol.json')
    best_epoch = int(summary['best_metric_epochs'][
        summary['selection_metric']])
    validation = _read_json(_resolve_validation_diagnostics(
        run_dir, best_epoch))
    final_test = _read_json(run_dir / 'final_test_results.json', required=False)
    metrics = validation['metrics']

    raw_ade = metrics.get('Raw_minADE')
    aligned_ade = metrics.get('Aligned_minADE')
    raw_fde = metrics.get('Raw_minFDE')
    aligned_fde = metrics.get('Aligned_minFDE')
    preserved = (
        abs((raw_ade or 0) - (aligned_ade or 0)) <= 1e-6 and
        abs((raw_fde or 0) - (aligned_fde or 0)) <= 1e-6 and
        metrics.get('marginal_ADE_max_abs_error', 1) <= 1e-6 and
        metrics.get('marginal_FDE_max_abs_error', 1) <= 1e-6)
    large_bucket = validation.get('component_size_buckets', {}).get(
        'comp_size>10', {})
    large_failed = (
        large_bucket.get('count', 0) > 0 and
        large_bucket.get('JADE', 0) >= large_bucket.get('Raw_JADE', 0))
    stage1_recommendation = (
        '建议保留 Stage1 upstream 作为消融，但主结果仍应先采用更干净的 '
        'Stage0 strong-marginal + V4 coupling；只有同协议 Stage1+V4 明显更优时再切换。')
    next_ablation = (
        '优先在完全相同的 20 个 seeds 和内部验证划分上比较 '
        'Stage0/Stage1 upstream × no-coupler/V4；随后比较 lowrank/MLP energy '
        '与 trajectory-conditioned relation on/off。')

    validation_source = protocol.get('internal_validation_source')
    dataset_name = (
        Path(validation_source).parent.parent.name.upper()
        if validation_source else 'Dataset')

    lines = [
        f'# {dataset_name} Multiway Coupling V4 Results', '',
        '> 本文件由训练产物自动生成。模型选择只使用内部验证，held-out test '
        '仅在 best checkpoint 确定后评测。', '',
        '## 1. 最佳模型与协议', '',
        f"- selection metric：`{summary['selection_metric']}`",
        f'- best epoch：**{best_epoch}**',
        f"- model selection split：`{protocol['model_selection_split']}`",
        f"- upstream generator：`{protocol.get('upstream_generator')}`",
        f"- upstream checkpoint：`{_display_path(protocol.get('upstream_checkpoint'))}`",
        f"- upstream frozen：{_fmt(protocol.get('freeze_upstream_generator'))}",
        f"- internal validation strategy：`{protocol['internal_validation_strategy']}`",
        f"- internal validation source：`{_display_path(protocol.get('internal_validation_source'))}`",
        f"- train / valid / final-test windows：{protocol['train_window_count']} / "
        f"{protocol['valid_window_count']} / {protocol['test_window_count']}",
        f"- final test split：`{protocol['final_test_split']}`", '',
        '## 2. Best internal-validation 指标', '',
        '| 指标 | Raw | Aligned | Aligned - Raw |',
        '|---|---:|---:|---:|',
        f"| minADE | {_fmt(raw_ade)} | {_fmt(aligned_ade)} | "
        f"{_fmt((aligned_ade or 0) - (raw_ade or 0))} |",
        f"| minFDE | {_fmt(raw_fde)} | {_fmt(aligned_fde)} | "
        f"{_fmt((aligned_fde or 0) - (raw_fde or 0))} |",
        f"| JADE | {_fmt(metrics.get('Raw_JADE'))} | "
        f"{_fmt(metrics.get('JADE'))} | {_fmt(-metrics.get('delta_JADE', 0))} |",
        f"| JFDE | {_fmt(metrics.get('Raw_JFDE'))} | "
        f"{_fmt(metrics.get('JFDE'))} | {_fmt(-metrics.get('delta_JFDE', 0))} |",
        '',
        f'- Marginal preservation check：**{"通过" if preserved else "失败"}**',
        f"- max minADE absolute gap：{_fmt(metrics.get('marginal_ADE_max_abs_error'))}",
        f"- max minFDE absolute gap：{_fmt(metrics.get('marginal_FDE_max_abs_error'))}",
        f"- alignment oracle：{_fmt(metrics.get('alignment_oracle'))}",
        f"- alignment headroom：{_fmt(metrics.get('alignment_headroom'))}",
        f"- recovery ratio：{_fmt(metrics.get('alignment_recovery_ratio'))}", '',
        '## 3. Coupling 行为诊断', '',
        f"- changed fraction：{_fmt(metrics.get('changed_fraction'))}",
        f"- keep confidence：{_fmt(metrics.get('keep_confidence'))}",
        f"- mean pair gate：{_fmt(metrics.get('mean_pair_gate'))}",
        f"- relation entropy：{_fmt(metrics.get('relation_entropy'))}",
        f"- permutation entropy：{_fmt(metrics.get('permutation_entropy'))}",
        f"- Sinkhorn row / col error：{_fmt(metrics.get('sinkhorn_row_error'))} / "
        f"{_fmt(metrics.get('sinkhorn_col_error'))}",
        f"- soft-hard gap：{_fmt(metrics.get('soft_hard_gap'))}",
        f"- Hungarian / greedy JADE：{_fmt(metrics.get('JADE'))} / "
        f"{_fmt(metrics.get('greedy_JADE'))}",
        f"- Hungarian - greedy JADE：{_fmt(metrics.get('hungarian_minus_greedy_JADE'))}",
        f"- average N / edges / degree：{_fmt(metrics.get('avg_num_agents'), 2)} / "
        f"{_fmt(metrics.get('avg_num_edges'), 2)} / {_fmt(metrics.get('avg_degree'), 2)}",
        f"- maximum component size：{_fmt(metrics.get('max_component_size'), 0)}", '',
    ]
    lines += _bucket_table(
        '按 scene N 分桶', validation.get('N_size_buckets', {}))
    lines += [''] + _bucket_table(
        '按 connected-component size 分桶',
        validation.get('component_size_buckets', {}))
    lines += ['', '## 4. 结论与下一步', '',
              f'- 大 component 是否仍显著失效：**{"是" if large_failed else "当前未观察到"}**',
              f'- Stage1 upstream：{stage1_recommendation}',
              f'- 最值得做的下一步消融：{next_ablation}', '']

    if final_test is not None:
        lines += ['## 5. Held-out final test', '',
                  f"- checkpoint：`{final_test['checkpoint']}` "
                  f"(epoch {final_test['checkpoint_epoch']})", '']
        for name, value in final_test['average'].items():
            lines.append(f'- {name}: {_fmt(value)}')
        lines.append('')
    else:
        lines += ['## 5. Held-out final test', '',
                  '- 尚未产生 `final_test_results.json`；不会用 validation '
                  '数字冒充 final test。', '']

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote V4 report: {output}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=Path, required=True)
    parser.add_argument(
        '--output', type=Path,
        default=Path('docs/UNIV_MULTIWAY_COUPLING_V4_RESULTS.md'))
    args = parser.parse_args()
    generate(args.run_dir, args.output)


if __name__ == '__main__':
    main()
