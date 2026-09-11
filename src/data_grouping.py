"""Shared decisions for legacy versus synchronized scene-window batches."""

import os

SCENE_BATCH_FORMAT_VERSION = 2
CACHE_MANIFEST_VERSION = 1
CACHE_MANIFEST_FILENAME = 'cache_manifest.json'


def requires_scene_window_batches(args):
    """Return whether an ablation requires valid simultaneous-agent groups."""
    return getattr(args, 'goal_model_type', 'independent') != 'independent'


def batch_cache_dirname(args):
    """Keep the original cache untouched and version social-window caches."""
    if requires_scene_window_batches(args):
        dirname = f'data_batches_joint_v{SCENE_BATCH_FORMAT_VERSION}'
    else:
        dirname = 'data_batches'
    if getattr(args, 'compress_batch_cache', False):
        dirname += '_zstd_v1'
    return dirname


def batch_cache_path(args):
    """Return the cache path, optionally on a separate scratch volume.

    Large ETH/UCY trajectory-map caches can be tens of GiB per held-out
    scene. Keeping them separate from checkpoints lets concurrent runs use
    persistent and scratch volumes while logs and weights stay in ``output``.
    """
    cache_root = getattr(args, 'batch_cache_root', None)
    if cache_root:
        return os.path.join(
            os.path.abspath(os.path.expanduser(cache_root)),
            str(getattr(args, 'dataset', 'dataset')),
            str(getattr(args, 'test_set', 'test')),
            batch_cache_dirname(args))
    save_dir = args.save_dir
    # Named runs isolate configs/logs/checkpoints but reuse the exact same
    # immutable preprocessing cache from their parent model ablation.
    if getattr(args, 'run_name', None) is not None:
        save_dir = os.path.dirname(os.path.dirname(save_dir))
    return os.path.join(save_dir, batch_cache_dirname(args))


def batch_cache_manifest(args):
    """Return preprocessing settings whose changes invalidate cached batches."""
    manifest = {
        'manifest_version': CACHE_MANIFEST_VERSION,
        'batch_format_version': (
            SCENE_BATCH_FORMAT_VERSION
            if requires_scene_window_batches(args) else 1),
        'goal_model_type': getattr(args, 'goal_model_type', 'independent'),
        'dataset': getattr(args, 'dataset', None),
        'test_set': getattr(args, 'test_set', None),
        'obs_length': int(getattr(args, 'obs_length', 8)),
        'pred_length': int(getattr(args, 'pred_length', 12)),
        'seq_length': int(getattr(args, 'seq_length', 20)),
        'down_factor': int(getattr(args, 'down_factor', 8)),
        'skip_ts_window': int(getattr(args, 'skip_ts_window', 1)),
        'batch_size': int(getattr(args, 'batch_size', 64)),
        'shuffle_train_batches': bool(getattr(
            args, 'shuffle_train_batches', True)),
        'fast_debug': bool(getattr(args, 'fast_debug', False)),
        'fast_debug_num': int(getattr(args, 'fast_debug_num', 3)),
    }
    # Keep the legacy manifest byte-for-byte compatible when compression is
    # disabled; compressed caches already use a separate versioned directory.
    if getattr(args, 'compress_batch_cache', False):
        manifest['compression'] = 'zstd-v1'
    return manifest
