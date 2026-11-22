"""Small CPU contract checks using synthetic metadata only, never real truth."""
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from joint_flow_frozen_test import run, scoring


def test_actual_guard_rejects_future_before_reader_and_restores_reader(monkeypatch):
    reads = []

    def underlying(self, index, purpose, truth=True):
        reads.append((index, purpose, truth))
        return 'synthetic item'

    monkeypatch.setattr(run.FormalData, 'item', underlying)
    owner = object()
    with run.forbid_heldout_truth():
        with pytest.raises(RuntimeError, match='heldout truth forbidden'):
            run.FormalData.item(owner, 7, run.PURPOSE, truth=True)
        with pytest.raises(RuntimeError, match='heldout truth forbidden'):
            run.FormalData.item(owner, 8, run.PURPOSE)
        assert reads == []
        assert run.FormalData.item(owner, 9, 'selection_dev', truth=True) == 'synthetic item'
        assert run.FormalData.item(owner, 10, run.PURPOSE, truth=False) == 'synthetic item'
        assert reads == [(9, 'selection_dev', True), (10, run.PURPOSE, False)]
    assert run.FormalData.item is underlying


def test_five_seed_sample_sd_and_paired_difference():
    a = [1., 2., 4., 8., 16.]
    b = [2., 1., 6., 6., 16.]
    result = scoring._mean_std(a)
    assert result['mean'] == pytest.approx(6.2)
    assert result['std'] == pytest.approx(np.std(a, ddof=1))
    assert result['std'] != pytest.approx(np.std(a, ddof=0))
    paired = scoring._mean_std([y-x for x, y in zip(a, b)])
    assert paired['mean'] == 0.
    assert paired['std'] == pytest.approx(math.sqrt(10/4))
    assert paired['std'] != pytest.approx(scoring._mean_std(b)['std']-result['std'])
    with pytest.raises(ValueError, match='invalid seed statistics'):
        scoring._mean_std(a[:4])


@pytest.mark.parametrize('defect,reason', [
    ('not_persisted', 'seal must already be persisted'),
    ('incomplete_rows', 'incomplete five-seed seal'),
    ('missing_trajectory', 'sealed prediction hash mismatch'),
])
def test_invalid_seal_cannot_construct_scoring_reader(tmp_path, monkeypatch, defect, reason):
    monkeypatch.setattr(scoring, 'RUN', tmp_path)
    constructor_calls = []

    def denied_reader(*args, **kwargs):
        constructor_calls.append((args, kwargs))
        raise AssertionError('scoring data constructed before predictions were verified')

    monkeypatch.setattr(scoring, 'FormalData', denied_reader)
    indices = list(range(998))
    inputs = SimpleNamespace(identity={'synthetic': True}, split={
        'heldout_confirmation_indices': indices,
        'members': {index: {'agent_ids': ['synthetic-agent']} for index in indices},
    })
    (tmp_path/'RUN_CONFIG.json').write_text('{}')
    rows = []
    if defect == 'missing_trajectory':
        rows = [dict(seed=seed, index=index, n=1, window=f'test_{index+1:04d}',
                     file=f'predictions/seed_{seed}/{index}.npz', sha256='0'*64)
                for seed in scoring.SEEDS for index in indices]
    seal = dict(identity=inputs.identity,
                source_config_sha256=scoring.digest(tmp_path/'RUN_CONFIG.json'),
                windows=998, seeds=scoring.SEEDS, seal_unix=1., rows=rows)
    if defect != 'not_persisted':
        (tmp_path/'PREDICTIONS_FROZEN.json').write_text(json.dumps(seal))
    # Verify both the verification API and the public score entry point fail first.
    with pytest.raises(ValueError, match=reason):
        scoring.verify_seal(inputs, seal)
    with pytest.raises(ValueError, match=reason):
        scoring.score(inputs, seal)
    assert constructor_calls == []
    assert not (tmp_path/'PRE_SCORE_SEAL_VERIFICATION.json').exists()
