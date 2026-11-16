"""Cumulative read-only budget shared by frozen-flow and MDN export stages."""
import signal
import time
from contextlib import contextmanager
from bcep.pilot_runner import atomic_json
import json


@contextmanager
def readonly_budget(root, stage):
    path = root/'reports/READONLY_BUDGET.json'
    ledger = json.loads(path.read_text()) if path.exists() else dict(limit_seconds=1800, seconds=0., stages=[])
    if any(x['stage'] == stage for x in ledger['stages']):
        raise RuntimeError('read-only stage already attempted; no overwrite or reset')
    remaining = ledger['limit_seconds'] - ledger['seconds']
    if remaining <= 0: raise RuntimeError('READONLY_BUDGET_EXHAUSTED')
    started = time.monotonic()
    row = dict(stage=stage, status='RUNNING', seconds=0.)
    ledger['stages'].append(row); atomic_json(ledger, path)
    def expired(signum, frame): raise TimeoutError('READONLY_BUDGET_EXHAUSTED')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        yield
        row['status'] = 'COMPLETE'
    except BaseException as error:
        row.update(status='FAILED_OR_INCOMPLETE', error=repr(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)
        row['seconds'] = time.monotonic() - started; ledger['seconds'] += row['seconds']
        atomic_json(ledger, path)
