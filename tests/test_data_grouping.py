import pickle
import os
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from src.data_grouping import (
    SCENE_BATCH_FORMAT_VERSION,
    batch_cache_manifest,
    batch_cache_dirname,
    batch_cache_path,
    requires_scene_window_batches,
)


# ``data_pre_process`` eagerly imports every real-dataset adapter.  Pandas is
# optional in the lightweight AB3D test environment, so provide only the type
# name those adapters reference during import.  The tests below deliberately
# use _MiniFrame instead and never replace production preprocessing behavior.
try:
    import pandas  # noqa: F401
except ModuleNotFoundError:
    pandas_stub = types.ModuleType('pandas')
    pandas_stub.DataFrame = type('DataFrame', (), {})
    sys.modules['pandas'] = pandas_stub

from src.data_pre_process import Trajectory_Data_Pre_Process


class _ColumnILoc:
    def __init__(self, values):
        self._values = values

    def __getitem__(self, index):
        return self._values[index]


class _Column:
    def __init__(self, values):
        self._values = np.asarray(values)

    def __iter__(self):
        return iter(self._values)

    def __eq__(self, other):
        return self._values == other

    @property
    def iloc(self):
        return _ColumnILoc(self._values)

    @property
    def values(self):
        return self._values

    def to_numpy(self):
        return self._values.copy()


class _Matrix:
    def __init__(self, values):
        self.values = np.asarray(values)


class _FrameILoc:
    def __init__(self, frame):
        self._frame = frame

    def __getitem__(self, index):
        selected = self._frame._rows[index]
        if isinstance(selected, dict):
            return selected
        return _MiniFrame(selected)


class _MiniFrame:
    """Small DataFrame-shaped object covering the production grouping API."""

    def __init__(self, rows):
        self._rows = list(rows)

    def __len__(self):
        return len(self._rows)

    def __getattr__(self, name):
        if self._rows and name in self._rows[0]:
            return _Column([row[name] for row in self._rows])
        raise AttributeError(name)

    def __getitem__(self, key):
        if isinstance(key, (list, tuple)) and all(
                isinstance(column, str) for column in key):
            return _Matrix([
                [row[column] for column in key] for row in self._rows
            ])

        mask = np.asarray(key, dtype=bool)
        if mask.shape != (len(self._rows),):
            raise IndexError('boolean mask must match the number of rows')
        return _MiniFrame([
            row for row, keep in zip(self._rows, mask) if keep
        ])

    @property
    def iloc(self):
        return _FrameILoc(self)

    def sort_values(self, column):
        return _MiniFrame(sorted(self._rows, key=lambda row: row[column]))


@pytest.mark.parametrize(
    ('goal_model_type', 'expected_joint'),
    [
        ('independent', False),
        ('social', True),
        ('lowrank', True),
        ('energy', True),
        ('joint', True),
    ],
)
def test_batch_mode_selects_scene_windows_only_for_non_independent_models(
        goal_model_type, expected_joint):
    args = SimpleNamespace(goal_model_type=goal_model_type)

    assert requires_scene_window_batches(args) is expected_joint


def test_independent_mode_keeps_original_cache_directory():
    assert batch_cache_dirname(SimpleNamespace()) == 'data_batches'
    assert batch_cache_dirname(
        SimpleNamespace(goal_model_type='independent')) == 'data_batches'
    assert batch_cache_dirname(
        SimpleNamespace(goal_model_type='joint')) == (
            f'data_batches_joint_v{SCENE_BATCH_FORMAT_VERSION}')
    assert batch_cache_dirname(SimpleNamespace(
        goal_model_type='independent', compress_batch_cache=True)) == (
            'data_batches_zstd_v1')


def test_batch_cache_path_defaults_to_save_dir_and_supports_scratch_root(
        tmp_path):
    default_args = SimpleNamespace(
        save_dir='/models/eth', goal_model_type='independent')
    assert batch_cache_path(default_args) == '/models/eth/data_batches'

    scratch_args = SimpleNamespace(
        save_dir='/models/eth', batch_cache_root=str(tmp_path),
        dataset='eth5', test_set='hotel', goal_model_type='independent')
    assert batch_cache_path(scratch_args) == str(
        tmp_path / 'eth5' / 'hotel' / 'data_batches')

    named_run_args = SimpleNamespace(
        save_dir='/models/eth/joint/runs/optimized_v1',
        run_name='optimized_v1', goal_model_type='joint',
        compress_batch_cache=False, batch_cache_root=None)
    assert batch_cache_path(named_run_args) == \
        '/models/eth/joint/data_batches_joint_v2'


def test_identical_eval_split_reuses_batches_with_hard_links(tmp_path):
    valid_source = tmp_path / 'val' / 'scene.txt'
    test_source = tmp_path / 'test' / 'scene.txt'
    valid_source.parent.mkdir()
    test_source.parent.mkdir()
    valid_source.write_bytes(b'identical trajectory rows')
    test_source.write_bytes(b'identical trajectory rows')

    valid_batches = tmp_path / 'valid_batches'
    test_batches = tmp_path / 'test_batches'
    valid_batches.mkdir()
    test_batches.mkdir()
    valid_batch = valid_batches / 'valid_batch_0001.pkl'
    valid_batch.write_bytes(b'cached tensors')

    processor = Trajectory_Data_Pre_Process.__new__(
        Trajectory_Data_Pre_Process)
    processor.args = SimpleNamespace(shuffle_test_batches=False)
    processor.experiment = SimpleNamespace(data={
        'valid': [{
            'file_path': str(valid_source), 'scene_name': 'scene',
            'downsample_frame_rate': 1,
        }],
        'test': [{
            'file_path': str(test_source), 'scene_name': 'scene',
            'downsample_frame_rate': 1,
        }],
    })
    processor.batches_folders = {
        'valid': str(valid_batches), 'test': str(test_batches)}
    processor.batches_confirmation_files = {
        'test': str(tmp_path / 'finished_test_batches.txt')}
    processor.num_batches = {'test': 0}

    assert processor._link_identical_eval_batches() is True
    linked_batch = test_batches / 'test_batch_0001.pkl'
    assert linked_batch.read_bytes() == b'cached tensors'
    assert os.stat(valid_batch).st_ino == os.stat(linked_batch).st_ino
    assert processor.num_batches['test'] == 1


def test_scene_window_grouping_requires_identical_frame_ids_and_never_mixes(
        monkeypatch):
    rows = []
    frames_by_agent = {
        10: [0, 1, 2, 10, 11, 12],
        20: [0, 1, 2, 10, 11, 12],
        30: [0, 1, 2],
        40: [10, 11, 12],
    }
    for agent_id, frame_ids in frames_by_agent.items():
        for frame_id in frame_ids:
            rows.append({
                'frame_id': frame_id,
                'agent_id': agent_id,
                # Encoding the frame/agent in coordinates lets the capture
                # assert which original rows production code grouped.
                'x_coord': float(frame_id),
                'y_coord': float(agent_id),
            })

    processor = object.__new__(Trajectory_Data_Pre_Process)
    processor.args = SimpleNamespace(
        seq_length=3,
        skip_ts_window=3,
        shuffle_train_batches=False,
        fast_debug=False,
        fast_debug_num=100,
    )
    processor.dataset = SimpleNamespace(
        scenes={'synthetic': SimpleNamespace(delta_frame=1)})
    processor.num_batches = {'valid': 0}

    captured = []

    def capture_batch(self, trajectories, batch_ids, set_name,
                      frame_ids=None):
        captured.append({
            'trajectories': [trajectory.copy()
                             for trajectory in trajectories],
            'batch_ids': dict(batch_ids),
            'set_name': set_name,
            'frame_ids': frame_ids.copy(),
        })

    monkeypatch.setattr(
        processor,
        'massup_batch_and_save',
        types.MethodType(capture_batch, processor),
    )
    scene_data = {
        'scene_name': 'synthetic',
        'downsample_frame_rate': 1,
        # Reverse order ensures correctness does not depend on row ordering.
        'raw_pixel_data': _MiniFrame(reversed(rows)),
        'set_name': 'valid',
        'file_path': 'in-memory.csv',
    }

    processor.make_scene_window_batches(scene_data, 'valid')

    assert len(captured) == 2
    batches_by_frames = {
        tuple(batch['frame_ids']): batch for batch in captured
    }
    assert set(batches_by_frames) == {(0, 1, 2), (10, 11, 12)}
    assert set(batches_by_frames[(0, 1, 2)]['batch_ids']['agent_ids']) == {
        10, 20, 30,
    }
    assert set(batches_by_frames[(10, 11, 12)]['batch_ids']['agent_ids']) == {
        10, 20, 40,
    }

    for frame_key, batch in batches_by_frames.items():
        assert batch['set_name'] == 'valid'
        assert batch['batch_ids']['frame_ids'] == list(frame_key)
        assert batch['batch_ids']['starting_frames'] == [
            frame_key[0]
        ] * len(batch['trajectories'])
        assert batch['batch_ids']['synchronized_window'] is True
        assert batch['batch_ids']['batch_format_version'] == (
            SCENE_BATCH_FORMAT_VERSION)
        for trajectory in batch['trajectories']:
            # Every agent column has the full, exact frame sequence for this
            # window; no trajectory can contain rows from the other window.
            np.testing.assert_array_equal(trajectory[:, 0], frame_key)


def test_downsampling_uses_one_scene_phase_not_each_agents_entry_frame(
        monkeypatch):
    """Late entry must not shift an agent onto a false temporal phase."""
    rows = []
    for agent_id, frames in {10: range(0, 6), 20: range(1, 6)}.items():
        rows.extend({
            'frame_id': frame,
            'agent_id': agent_id,
            'x_coord': float(frame),
            'y_coord': float(agent_id),
        } for frame in frames)

    processor = object.__new__(Trajectory_Data_Pre_Process)
    processor.args = SimpleNamespace(
        seq_length=2, skip_ts_window=1,
        shuffle_train_batches=False, fast_debug=False,
        fast_debug_num=100)
    processor.dataset = SimpleNamespace(
        scenes={'synthetic': SimpleNamespace(delta_frame=1)})
    processor.num_batches = {'valid': 0}
    captured = []

    def capture(self, trajectories, batch_ids, set_name, frame_ids=None):
        captured.append((tuple(frame_ids), tuple(batch_ids['agent_ids'])))

    monkeypatch.setattr(
        processor, 'massup_batch_and_save',
        types.MethodType(capture, processor))
    processor.make_scene_window_batches({
        'scene_name': 'synthetic',
        'downsample_frame_rate': 2,
        'raw_pixel_data': _MiniFrame(rows),
        'set_name': 'valid',
        'file_path': 'in-memory.csv',
    }, 'valid')

    # Agent 20 enters on odd frame 1, but both tracks are aligned to the
    # scene's even grid. They therefore correctly share window (2,4).
    assert ((2, 4), (10, 20)) in captured


def test_window_stride_uses_scene_grid_not_each_agents_entry_row(monkeypatch):
    rows = []
    for agent_id, frames in {10: range(0, 7), 20: range(1, 7)}.items():
        rows.extend({
            'frame_id': frame,
            'agent_id': agent_id,
            'x_coord': float(frame),
            'y_coord': float(agent_id),
        } for frame in frames)

    processor = object.__new__(Trajectory_Data_Pre_Process)
    processor.args = SimpleNamespace(
        seq_length=3, skip_ts_window=2,
        shuffle_train_batches=False, fast_debug=False,
        fast_debug_num=100)
    processor.dataset = SimpleNamespace(
        scenes={'synthetic': SimpleNamespace(delta_frame=1)})
    processor.num_batches = {'valid': 0}
    captured = []

    def capture(self, trajectories, batch_ids, set_name, frame_ids=None):
        captured.append((tuple(frame_ids), tuple(batch_ids['agent_ids'])))

    monkeypatch.setattr(
        processor, 'massup_batch_and_save',
        types.MethodType(capture, processor))
    processor.make_scene_window_batches({
        'scene_name': 'synthetic',
        'downsample_frame_rate': 1,
        'raw_pixel_data': _MiniFrame(rows),
        'set_name': 'valid',
        'file_path': 'in-memory.csv',
    }, 'valid')

    assert ((2, 3, 4), (10, 20)) in captured


def test_cache_manifest_changes_for_every_preprocessing_sensitive_setting():
    base = SimpleNamespace(
        goal_model_type='joint', dataset='sdd', test_set='sdd',
        obs_length=8, pred_length=12, seq_length=20, down_factor=8,
        skip_ts_window=1, batch_size=64, shuffle_train_batches=True,
        fast_debug=False, fast_debug_num=3)
    reference = batch_cache_manifest(base)
    for key, replacement in {
            'down_factor': 4, 'skip_ts_window': 2, 'batch_size': 16,
            'fast_debug': True, 'fast_debug_num': 1}.items():
        changed = SimpleNamespace(**vars(base))
        setattr(changed, key, replacement)
        assert batch_cache_manifest(changed) != reference


def _bare_writer(tmp_path):
    processor = object.__new__(Trajectory_Data_Pre_Process)
    processor.num_batches = {'valid': 0}
    processor.batches_folders = {'valid': str(tmp_path)}

    def skip_cnn_maps(self, data_dict, batch_ids):
        return data_dict

    processor.add_pre_computed_cnn_maps = types.MethodType(
        skip_cnn_maps, processor)
    return processor


def _read_only_batch(tmp_path):
    batch_paths = list(tmp_path.glob('valid_batch_*.pkl'))
    assert len(batch_paths) == 1
    with batch_paths[0].open('rb') as handle:
        return pickle.load(handle)


def test_saved_scene_window_has_consistent_scene_index_ptr_and_frame_matrix(
        tmp_path):
    processor = _bare_writer(tmp_path)
    trajectories = [
        np.asarray([[4.0, 1.0], [6.0, 1.0], [8.0, 1.0]]),
        np.asarray([[4.0, 2.0], [6.0, 2.0], [8.0, 2.0]]),
        np.asarray([[4.0, 3.0], [6.0, 3.0], [8.0, 3.0]]),
    ]
    batch_ids = {
        'scene_name': 'synthetic',
        'frame_ids': [4, 6, 8],
        'agent_ids': [1, 2, 3],
        'synchronized_window': True,
        'batch_format_version': SCENE_BATCH_FORMAT_VERSION,
    }

    processor.massup_batch_and_save(
        trajectories,
        batch_ids,
        'valid',
        frame_ids=np.asarray([4, 6, 8], dtype=np.int64),
    )

    data_dict, saved_batch_ids = _read_only_batch(tmp_path)
    assert data_dict['abs_pixel_coord'].shape == (3, 3, 2)
    np.testing.assert_array_equal(data_dict['scene_index'], [0, 0, 0])
    np.testing.assert_array_equal(data_dict['scene_ptr'], [0, 3])
    np.testing.assert_array_equal(
        data_dict['frame_ids'],
        np.asarray([[4, 4, 4], [6, 6, 6], [8, 8, 8]]),
    )
    assert int(data_dict['batch_format_version']) == (
        SCENE_BATCH_FORMAT_VERSION)
    assert saved_batch_ids == batch_ids


def test_legacy_writer_omits_joint_metadata(tmp_path):
    processor = _bare_writer(tmp_path)
    trajectories = [
        np.asarray([[0.0, 1.0], [1.0, 1.0]]),
        np.asarray([[10.0, 2.0], [11.0, 2.0]]),
    ]
    legacy_batch_ids = {
        'scene_name': 'synthetic',
        'starting_frames': [0, 10],
        'agent_ids': [1, 2],
    }

    processor.massup_batch_and_save(
        trajectories, legacy_batch_ids, 'valid')

    data_dict, saved_batch_ids = _read_only_batch(tmp_path)
    assert set(data_dict) == {'abs_pixel_coord', 'seq_list'}
    assert saved_batch_ids == legacy_batch_ids
