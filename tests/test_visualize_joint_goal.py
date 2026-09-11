import os
from pathlib import Path
import subprocess
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from tools.visualize_joint_goal import plot_joint_goal_scene


def _structured_scene():
    """Return one labelled co-walking/crossing diagnostic scene."""
    return {
        'obs': np.array([
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.5, 1.0], [1.0, 1.0]],
            [[2.0, -1.0], [2.0, -0.5], [2.0, 0.0]],
        ], dtype=np.float32),
        'gt': np.array([
            [[1.5, 0.0], [2.0, 0.0]],
            [[1.5, 1.0], [2.0, 1.0]],
            [[2.0, 0.5], [2.0, 1.0]],
        ], dtype=np.float32),
        'independent_goals': np.array([
            [[2.0, 0.0], [2.2, 0.2]],
            [[2.0, 1.0], [2.2, 1.2]],
            [[2.0, 1.0], [2.3, 0.8]],
        ], dtype=np.float32),
        'independent_prob': np.array([
            [0.8, 0.2], [0.7, 0.3], [0.6, 0.4],
        ], dtype=np.float32),
        'joint_goals': np.array([
            [[2.0, 0.0], [2.2, 0.1]],
            [[2.0, 1.0], [2.2, 1.1]],
            [[2.0, 1.0], [2.4, 0.9]],
        ], dtype=np.float32),
        'predictions': np.array([
            [
                [[1.5, 0.0], [1.5, 1.0], [2.0, 0.5]],
                [[2.0, 0.0], [2.0, 1.0], [2.0, 1.0]],
            ],
            [
                [[1.6, 0.05], [1.6, 1.05], [2.2, 0.45]],
                [[2.2, 0.1], [2.2, 1.1], [2.4, 0.9]],
            ],
        ], dtype=np.float32),
        'edge_index': np.array([
            [0, 0, 1],
            [1, 2, 2],
        ], dtype=np.int64),
        'relation_prob': np.array([
            [0.85, 0.05, 0.05, 0.05],
            [0.80, 0.05, 0.10, 0.05],
            [0.05, 0.80, 0.10, 0.05],
        ], dtype=np.float32),
        'mode_prob': np.array([0.35, 0.65], dtype=np.float32),
        'selected_mode': np.array([0, 1], dtype=np.int64),
    }


def test_plot_uses_agg_and_renders_structured_annotations():
    assert mpl.get_backend().lower() == 'agg'

    fig = plot_joint_goal_scene(
        _structured_scene(), sample_index=1, coordinates='world',
        annotate_relations=True)
    try:
        assert len(fig.axes) == 1
        axis = fig.axes[0]
        assert axis.get_xlabel() == 'x (m)'
        assert axis.get_ylabel() == 'y (m)'
        assert 'social mode z=1' in axis.get_title(loc='left')
        assert sum(text.get_text().startswith('r') for text in axis.texts) == 3
        legend_labels = axis.get_legend_handles_labels()[1]
        assert {
            'Observed', 'Ground truth', 'Independent candidates',
            'Predicted future', 'Joint selected goals',
        }.issubset(set(legend_labels))
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_cli_generates_pdf_svg_tiff_and_png(tmp_path):
    archive_path = tmp_path / 'labelled_scene.npz'
    np.savez(archive_path, **_structured_scene())
    output_prefix = tmp_path / 'rendered' / 'joint_scene'
    repository_root = Path(__file__).resolve().parents[1]
    script_path = repository_root / 'tools' / 'visualize_joint_goal.py'
    environment = os.environ.copy()
    environment['MPLBACKEND'] = 'Agg'

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            str(archive_path),
            '--output-prefix', str(output_prefix),
            '--sample-index', '1',
            '--coordinates', 'world',
            '--annotate-relations',
        ],
        cwd=str(repository_root),
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    generated = {
        suffix: Path(str(output_prefix) + suffix)
        for suffix in ('.pdf', '.svg', '.tiff', '.png')
    }
    for path in generated.values():
        assert path.is_file()
        assert path.stat().st_size > 0

    assert generated['.pdf'].read_bytes().startswith(b'%PDF')
    assert b'<svg' in generated['.svg'].read_bytes()[:1000]
    assert generated['.png'].read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert generated['.tiff'].read_bytes()[:4] in (b'II*\x00', b'MM\x00*')


@pytest.mark.parametrize(
    ('key', 'invalid_value', 'exception'),
    [
        ('obs', np.zeros((3, 2), dtype=np.float32), ValueError),
        ('gt', np.zeros((2, 2, 2), dtype=np.float32), ValueError),
        ('independent_goals', np.zeros((3, 2, 3), dtype=np.float32),
         ValueError),
        ('joint_goals', np.zeros((4, 2, 2), dtype=np.float32), ValueError),
        ('predictions', np.zeros((2, 2, 4, 2), dtype=np.float32), ValueError),
    ],
)
def test_plot_rejects_incompatible_array_shapes(key, invalid_value, exception):
    data = _structured_scene()
    data[key] = invalid_value

    with pytest.raises(exception):
        plot_joint_goal_scene(data)
    plt.close('all')


def test_plot_requires_observations_and_valid_sample_index():
    data = _structured_scene()
    data.pop('obs')
    with pytest.raises(KeyError, match='obs'):
        plot_joint_goal_scene(data)

    with pytest.raises(IndexError, match='sample_index'):
        plot_joint_goal_scene(_structured_scene(), sample_index=2)
    with pytest.raises(IndexError, match='non-negative'):
        plot_joint_goal_scene({'obs': _structured_scene()['obs']},
                              sample_index=-1)

    mode_only = {
        'obs': _structured_scene()['obs'],
        'selected_mode': np.array([0, 1], dtype=np.int64),
    }
    with pytest.raises(IndexError, match='sample_index'):
        plot_joint_goal_scene(mode_only, sample_index=2)
    plt.close('all')


def test_missing_relation_probability_is_not_fabricated():
    data = _structured_scene()
    data.pop('relation_prob')
    fig = plot_joint_goal_scene(data, annotate_relations=True)
    try:
        edge_labels = [text.get_text() for text in fig.axes[0].texts
                       if text.get_text() == 'edge' or
                       text.get_text().startswith('r')]
        assert edge_labels == ['edge'] * data['edge_index'].shape[1]
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ('key', 'invalid_value', 'message'),
    [
        ('predictions', np.full((1, 2, 3, 2), np.nan), 'NaN or Inf'),
        ('independent_prob', np.zeros((3, 2)), 'positive mass'),
        ('relation_prob', np.ones((3, 4)), 'sum to one'),
        ('mode_prob', np.array([0.2, 0.2]), 'sum to one'),
        ('edge_index', np.array([[0], [-1]]), 'missing agent'),
        ('edge_index', np.array([[1], [0]]), 'source < target'),
        ('selected_mode', np.array([0]), 'one entry per joint sample'),
        ('selected_mode', np.array([0, 2]), 'out-of-range'),
    ],
)
def test_plot_rejects_invalid_scientific_values(key, invalid_value, message):
    data = _structured_scene()
    data[key] = invalid_value
    with pytest.raises((ValueError, IndexError), match=message):
        plot_joint_goal_scene(data)
    plt.close('all')


def test_plot_requires_prediction_and_goal_sample_axes_to_match():
    data = _structured_scene()
    data['predictions'] = data['predictions'][:1]
    with pytest.raises(ValueError, match='same sample axis'):
        plot_joint_goal_scene(data)
    plt.close('all')
