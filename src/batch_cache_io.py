"""Read and write regular or losslessly compressed batch-cache pickles."""

import os
import pickle
import shutil
import subprocess
import sys


PICKLE_SUFFIX = '.pkl'
ZSTD_PICKLE_SUFFIX = '.pkl.zst'


def is_batch_cache_file(filename):
    return filename.endswith((PICKLE_SUFFIX, ZSTD_PICKLE_SUFFIX))


def batch_cache_files(folder):
    """Return deterministic paths for both supported cache formats."""
    return sorted(
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if is_batch_cache_file(name))


def _zstd_executable():
    candidates = [
        shutil.which('zstd'),
        os.path.join(os.path.dirname(sys.executable), 'zstd'),
    ]
    executable = next(
        (candidate for candidate in candidates
         if candidate and os.path.isfile(candidate) and
         os.access(candidate, os.X_OK)), None)
    if executable is None:
        raise RuntimeError(
            'Compressed batch caches require the zstd executable on PATH.')
    return executable


def dump_batch_cache(value, path, compressed=False):
    """Serialize one cache item, streaming through zstd when requested."""
    if not compressed:
        with open(path, 'wb') as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return

    temporary_path = path + '.tmp'
    with open(temporary_path, 'wb') as output_handle:
        process = subprocess.Popen(
            [_zstd_executable(), '-1', '-q', '-c'],
            stdin=subprocess.PIPE,
            stdout=output_handle,
            stderr=subprocess.PIPE)
        try:
            pickle.dump(
                value, process.stdin, protocol=pickle.HIGHEST_PROTOCOL)
            process.stdin.close()
            error_output = process.stderr.read()
            return_code = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
            raise
    if return_code != 0:
        os.remove(temporary_path)
        raise RuntimeError(
            f'zstd compression failed for {path}: '
            f'{error_output.decode(errors="replace")}')
    os.replace(temporary_path, path)


def load_batch_cache(path):
    """Deserialize one regular pickle or zstd-compressed pickle stream."""
    if path.endswith(PICKLE_SUFFIX):
        with open(path, 'rb') as handle:
            return pickle.load(handle)
    if not path.endswith(ZSTD_PICKLE_SUFFIX):
        raise ValueError(f'Unsupported batch cache file: {path}')

    process = subprocess.Popen(
        [_zstd_executable(), '-q', '-d', '-c', path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    try:
        value = pickle.load(process.stdout)
        process.stdout.close()
        error_output = process.stderr.read()
        return_code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    if return_code != 0:
        raise RuntimeError(
            f'zstd decompression failed for {path}: '
            f'{error_output.decode(errors="replace")}')
    return value
