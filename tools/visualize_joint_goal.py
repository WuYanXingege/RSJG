#!/usr/bin/env python3
"""Render a source-grounded diagnostic for one synchronized scene window.

Figure contract
---------------
Core conclusion: show whether one scene-level joint sample coordinates all
agents' endpoints while preserving formation or resolving crossing ambiguity.
The single spatial panel combines the evidence needed for that check: observed
and ground-truth motion, independent candidates, joint selections, sparse
interaction edges, latent relation confidence, and the selected social mode.

The input is an ``.npz`` file. Required key: ``obs [N,T_obs,2]``. Optional
keys are ``gt [N,T_fut,2]``, ``independent_goals [N,K,2]``,
``independent_prob [N,K]``, ``joint_goals [N,S,2]``,
``predictions [S,T,N,2]``, ``edge_index [2,E]``,
``relation_prob [E,M]``, ``mode_prob [R]``, and
``selected_mode [S]``. This tool never invents missing scientific values.
"""

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    'observed': '#263238',
    'ground_truth': '#2878B5',
    'prediction': '#E07A3F',
    'independent': '#9AA0A6',
    'joint': '#B33F62',
}
RELATION_COLORS = [
    '#4C78A8', '#F2A541', '#59A14F', '#B279A2', '#E15759', '#76B7B2'
]


def _configure_style():
    """Apply a restrained, editable-text scientific figure style."""
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
        'font.size': 7,
        'axes.labelsize': 7,
        'axes.titlesize': 8,
        'legend.fontsize': 6,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'axes.linewidth': 0.8,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'legend.frameon': False,
        'pdf.fonttype': 42,
        'svg.fonttype': 'none',
    })


def _validate_scene(data):
    """Validate array shapes; return ``(num_agents, sample_count)``."""
    if 'obs' not in data:
        raise KeyError("input archive must contain 'obs [N,T_obs,2]'")
    obs = np.asarray(data['obs'])
    if obs.ndim != 3 or obs.shape[-1] != 2:
        raise ValueError('obs must have shape [N,T_obs,2]')
    if not np.isfinite(obs).all():
        raise ValueError('obs contains NaN or Inf')
    num_agents = obs.shape[0]
    if num_agents == 0 or obs.shape[1] == 0:
        raise ValueError('obs must contain at least one agent and one time step')
    shape_specs = {
        'gt': (3, num_agents),
        'independent_goals': (3, num_agents),
        'joint_goals': (3, num_agents),
    }
    for key, (ndim, first_dim) in shape_specs.items():
        if key in data:
            value = np.asarray(data[key])
            if value.ndim != ndim or value.shape[0] != first_dim or \
                    value.shape[-1] != 2:
                raise ValueError(f'{key} has an incompatible shape {value.shape}')
            if key in {'independent_goals', 'joint_goals'} and value.shape[1] == 0:
                raise ValueError(f'{key} must contain at least one candidate/sample')
            if not np.isfinite(value).all():
                raise ValueError(f'{key} contains NaN or Inf')
    if 'independent_prob' in data:
        if 'independent_goals' not in data:
            raise ValueError('independent_prob requires independent_goals')
        value = np.asarray(data['independent_prob'])
        expected = np.asarray(data['independent_goals']).shape[:2]
        if value.shape != expected:
            raise ValueError(
                f'independent_prob must have shape {expected}, got {value.shape}')
        if not np.isfinite(value).all() or (value < 0).any():
            raise ValueError('independent_prob must be finite and non-negative')
        if (value.sum(axis=1) <= 0).any():
            raise ValueError('each independent_prob row must have positive mass')
    sample_count = None
    if 'joint_goals' in data:
        sample_count = np.asarray(data['joint_goals']).shape[1]
    if 'predictions' in data:
        value = np.asarray(data['predictions'])
        if value.ndim != 4 or value.shape[2] != num_agents or value.shape[-1] != 2:
            raise ValueError('predictions must have shape [S,T,N,2]')
        if value.shape[0] == 0 or value.shape[1] == 0:
            raise ValueError('predictions must contain a sample and future step')
        if not np.isfinite(value).all():
            raise ValueError('predictions contains NaN or Inf')
        if sample_count is not None and value.shape[0] != sample_count:
            raise ValueError(
                'predictions and joint_goals must share the same sample axis')
        sample_count = value.shape[0]
    if 'edge_index' in data:
        edge_index = np.asarray(data['edge_index'])
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError('edge_index must have shape [2,E]')
        if not np.issubdtype(edge_index.dtype, np.integer):
            if not np.isfinite(edge_index).all() or \
                    not np.equal(edge_index, np.floor(edge_index)).all():
                raise ValueError('edge_index must contain integer agent indices')
        if edge_index.size and ((edge_index < 0).any() or
                                (edge_index >= num_agents).any()):
            raise IndexError('edge_index refers to a missing agent')
        if edge_index.shape[1] and not np.all(edge_index[0] < edge_index[1]):
            raise ValueError(
                'edge_index must contain canonical source < target pairs')
        if edge_index.shape[1] and np.unique(
                edge_index, axis=1).shape[1] != edge_index.shape[1]:
            raise ValueError('edge_index contains duplicate pairs')
        if 'relation_prob' in data:
            value = np.asarray(data['relation_prob'])
            if value.ndim != 2 or value.shape[0] != edge_index.shape[1] or \
                    value.shape[1] == 0:
                raise ValueError('relation_prob must have shape [E,M] with M > 0')
            if not np.isfinite(value).all() or (value < 0).any():
                raise ValueError('relation_prob must be finite and non-negative')
            if value.shape[0] and not np.allclose(
                    value.sum(axis=1), 1.0, rtol=1e-4, atol=1e-6):
                raise ValueError('relation_prob rows must sum to one')
    elif 'relation_prob' in data:
        raise ValueError('relation_prob requires edge_index')
    if 'mode_prob' in data:
        value = np.asarray(data['mode_prob']).reshape(-1)
        if value.size == 0 or not np.isfinite(value).all() or (value < 0).any():
            raise ValueError('mode_prob must be a finite non-negative vector')
        if not np.isclose(value.sum(), 1.0, rtol=1e-4, atol=1e-6):
            raise ValueError('mode_prob must sum to one')
    if 'selected_mode' in data:
        value = np.asarray(data['selected_mode']).reshape(-1)
        if value.size == 0 or not np.issubdtype(value.dtype, np.integer) or \
                (value < 0).any():
            raise ValueError('selected_mode must contain non-negative integers')
        if sample_count is not None and value.size != sample_count:
            raise ValueError(
                'selected_mode must have one entry per joint sample')
        if sample_count is None:
            sample_count = value.size
        if 'mode_prob' in data and (value >= np.asarray(
                data['mode_prob']).reshape(-1).size).any():
            raise ValueError('selected_mode contains an out-of-range mode id')
    return num_agents, sample_count


def plot_joint_goal_scene(data, sample_index=0, coordinates='world',
                          annotate_relations=False):
    """Create the one-panel scene diagnostic and return the Matplotlib figure."""
    _configure_style()
    if sample_index < 0:
        raise IndexError('sample_index must be non-negative')
    if coordinates not in {'world', 'pixel'}:
        raise ValueError("coordinates must be either 'world' or 'pixel'")
    num_agents, sample_count = _validate_scene(data)
    if sample_count is not None and sample_index >= sample_count:
        raise IndexError('sample_index is outside the available sample axis')
    obs = np.asarray(data['obs'])
    # 180 x 120 mm, expressed directly in inches so static figure validators
    # can recover the intended publication width.
    fig, ax = plt.subplots(figsize=(7.086614, 4.724409),
                           constrained_layout=True)

    for agent in range(num_agents):
        label = 'Observed' if agent == 0 else None
        ax.plot(obs[agent, :, 0], obs[agent, :, 1], color=PALETTE['observed'],
                linewidth=1.2, marker='o', markersize=2.0, label=label, zorder=4)
        ax.text(obs[agent, -1, 0], obs[agent, -1, 1], f' {agent}',
                fontsize=6, color=PALETTE['observed'], va='bottom', zorder=7)

    if 'gt' in data:
        gt = np.asarray(data['gt'])
        for agent in range(num_agents):
            path = np.vstack((obs[agent, -1], gt[agent]))
            ax.plot(path[:, 0], path[:, 1], color=PALETTE['ground_truth'],
                    linewidth=1.0, linestyle='--',
                    label='Ground truth' if agent == 0 else None, zorder=3)

    if 'independent_goals' in data:
        goals = np.asarray(data['independent_goals'])
        probs = np.asarray(data['independent_prob']) \
            if 'independent_prob' in data else np.ones(goals.shape[:2])
        probs = probs / np.maximum(probs.max(axis=1, keepdims=True), 1e-12)
        for agent in range(num_agents):
            ax.scatter(goals[agent, :, 0], goals[agent, :, 1], marker='x',
                       s=10 + 12 * probs[agent], color=PALETTE['independent'],
                       alpha=0.35 + 0.45 * probs[agent], linewidths=0.7,
                       label='Independent candidates' if agent == 0 else None,
                       zorder=2)

    if 'predictions' in data:
        predictions = np.asarray(data['predictions'])
        if not 0 <= sample_index < predictions.shape[0]:
            raise IndexError('sample_index is outside predictions sample axis')
        for agent in range(num_agents):
            path = np.vstack((obs[agent, -1], predictions[sample_index, :, agent]))
            ax.plot(path[:, 0], path[:, 1], color=PALETTE['prediction'],
                    linewidth=1.0,
                    label='Predicted future' if agent == 0 else None, zorder=5)

    if 'joint_goals' in data:
        goals = np.asarray(data['joint_goals'])
        if not 0 <= sample_index < goals.shape[1]:
            raise IndexError('sample_index is outside joint_goals sample axis')
        selected = goals[:, sample_index]
        ax.scatter(selected[:, 0], selected[:, 1], marker='*', s=45,
                   color=PALETTE['joint'], edgecolor='white', linewidth=0.4,
                   label='Joint selected goals', zorder=8)

    if 'edge_index' in data:
        edge_index = np.asarray(data['edge_index'], dtype=int)
        relation_prob = np.asarray(data['relation_prob']) \
            if 'relation_prob' in data else None
        for edge, (source, target) in enumerate(edge_index.T):
            mode, confidence = None, None
            if relation_prob is not None:
                mode = int(relation_prob[edge].argmax())
                confidence = float(relation_prob[edge, mode])
            points = obs[[source, target], -1]
            color = (PALETTE['independent'] if mode is None else
                     RELATION_COLORS[mode % len(RELATION_COLORS)])
            ax.plot(points[:, 0], points[:, 1], color=color,
                    linewidth=(0.8 if confidence is None else
                               0.5 + confidence), alpha=0.75, zorder=1)
            if annotate_relations:
                midpoint = points.mean(axis=0)
                label = ('edge' if mode is None else
                         f'r{mode} {confidence:.2f}')
                ax.text(midpoint[0], midpoint[1], label,
                        fontsize=5, color=color, ha='center', va='bottom')

    title = f'Joint-goal scene sample {sample_index}'
    if 'selected_mode' in data:
        selected_modes = np.asarray(data['selected_mode']).reshape(-1)
        title += f' · social mode z={int(selected_modes[sample_index])}'
    elif 'mode_prob' in data:
        mode_prob = np.asarray(data['mode_prob']).reshape(-1)
        title += f' · MAP social mode z={int(mode_prob.argmax())}'
    ax.set_title(title, loc='left', fontweight='bold')
    unit = 'm' if coordinates == 'world' else 'px'
    ax.set_xlabel(f'x ({unit})')
    ax.set_ylabel(f'y ({unit})')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(color='#E6E8EB', linewidth=0.45, zorder=0)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='best', ncol=2)
    return fig


def save_figure_bundle(fig, output_prefix):
    """Export editable vector files plus high-resolution raster previews."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(prefix) + '.svg', bbox_inches='tight')
    fig.savefig(str(prefix) + '.pdf', bbox_inches='tight')
    fig.savefig(str(prefix) + '.tiff', dpi=600, bbox_inches='tight')
    fig.savefig(str(prefix) + '.png', dpi=300, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='One synchronized scene .npz')
    parser.add_argument('--output-prefix', required=True, type=Path)
    parser.add_argument('--sample-index', default=0, type=int)
    parser.add_argument('--coordinates', choices=['world', 'pixel'],
                        default='world')
    parser.add_argument('--annotate-relations', action='store_true')
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    fig = plot_joint_goal_scene(
        data, sample_index=args.sample_index, coordinates=args.coordinates,
        annotate_relations=args.annotate_relations)
    save_figure_bundle(fig, args.output_prefix)
    plt.close(fig)


if __name__ == '__main__':
    main()
