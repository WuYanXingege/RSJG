"""Execution-only switches, resolved once at import before a source-bound run.

The benchmark may disable a rejected optimization without changing a model,
seed, loss, or selector. The resolved list belongs in the run configuration.
"""
import os

NAMES = frozenset(('modulation','proposal','schedule','hungarian','geometry','logging','head_cache'))
DISABLED = frozenset(x.strip() for x in os.environ.get('RSJG_DISABLE_OPTIMIZATIONS','').split(',') if x.strip())
if not DISABLED <= NAMES:
    raise ValueError('unknown RSJG_DISABLE_OPTIMIZATIONS: '+repr(sorted(DISABLED-NAMES)))


def enabled(name):
    return name not in DISABLED


def policy():
    return {name:enabled(name) for name in sorted(NAMES)}
