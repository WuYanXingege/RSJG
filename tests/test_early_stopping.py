import pytest

from ebjd.early_stopping import EligibleJADEEarlyStop, RelativeEarlyStop


def relative(**overrides):
    values = dict(
        monitor="valid_ADE", earliest_stop_epoch=5, patience=3,
        relative_improvement=0.001)
    values.update(overrides)
    return RelativeEarlyStop(**values)


def test_true_best_and_significant_reference_are_independent():
    state = relative()
    assert state.observe(1.0, 1)["true_best_improved"]
    first = state.observe(0.9995, 2)
    assert first["true_best_improved"]
    assert not first["significant_improved"]
    assert state.best_score == pytest.approx(0.9995)
    assert state.significant_best == pytest.approx(1.0)
    second = state.observe(0.9989, 3)
    assert second["significant_improved"]
    assert state.significant_best == pytest.approx(0.9989)
    assert state.stale_validations == 0


def test_minimum_epoch_blocks_but_does_not_clear_patience():
    state = relative(earliest_stop_epoch=5, patience=2)
    state.observe(1.0, 1)
    assert not state.observe(1.0, 2)["should_stop"]
    assert not state.observe(1.0, 3)["should_stop"]
    assert state.stale_validations == 2
    assert state.observe(1.0, 5)["should_stop"]


def test_resume_preserves_all_counters():
    state = relative()
    state.observe(1.0, 1)
    state.observe(0.9999, 2)
    restored = RelativeEarlyStop.from_state_dict(state.state_dict())
    assert restored.state_dict() == state.state_dict()
    assert restored.observe(1.1, 5)["stale_validations"] == 2


def test_nonfinite_score_is_an_error():
    with pytest.raises(FloatingPointError):
        relative().observe(float("nan"), 1)


def test_eligible_phase_resets_patience_and_ineligible_counts_stale():
    monitor = EligibleJADEEarlyStop(relative(
        monitor="H_JADE", earliest_stop_epoch=2, patience=2))
    monitor.observe(1.0, False, 1)
    before = monitor.observe(1.0, False, 2)
    assert not before["entered_eligible_phase"]
    entered = monitor.observe(1.2, True, 3)
    assert entered["entered_eligible_phase"]
    assert monitor.eligible_phase
    assert monitor.state.stale_validations == 0
    assert not monitor.observe(0.5, False, 4)["should_stop"]
    assert monitor.observe(0.4, False, 5)["should_stop"]
    assert monitor.state.best_score == pytest.approx(1.2)


def test_eligible_monitor_resume_is_exact():
    monitor = EligibleJADEEarlyStop(relative(monitor="H_JADE"))
    monitor.observe(1.0, True, 4)
    monitor.observe(1.1, False, 5)
    restored = EligibleJADEEarlyStop.from_state_dict(monitor.state_dict())
    assert restored.state_dict() == monitor.state_dict()
