"""Checkpointable relative-improvement early stopping.

The exact historical best and the improvement reference used for patience are
intentionally independent.  Tiny improvements are still eligible for model
selection, while patience only resets after their cumulative improvement from
the last significant reference reaches the configured relative threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class RelativeEarlyStop:
    monitor: str
    earliest_stop_epoch: int
    patience: int
    relative_improvement: float
    best_score: float | None = None
    significant_best: float | None = None
    stale_validations: int = 0

    def __post_init__(self) -> None:
        if self.earliest_stop_epoch < 1:
            raise ValueError("earliest_stop_epoch must be positive")
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if not 0.0 < self.relative_improvement < 1.0:
            raise ValueError("relative_improvement must be in (0, 1)")

    def observe(self, score: float, completed_epoch: int) -> dict:
        score = float(score)
        if not math.isfinite(score):
            raise FloatingPointError(
                f"non-finite early-stop score for {self.monitor}: {score}")
        if completed_epoch < 1:
            raise ValueError("completed_epoch must be positive")
        true_best_improved = self.best_score is None or score < self.best_score
        if true_best_improved:
            self.best_score = score
        significant_improved = False
        if self.significant_best is None:
            self.significant_best = score
            self.stale_validations = 0
            significant_improved = True
        else:
            delta = max(
                abs(self.significant_best) * self.relative_improvement, 1e-12)
            if score < self.significant_best - delta:
                self.significant_best = score
                self.stale_validations = 0
                significant_improved = True
            else:
                self.stale_validations += 1
        should_stop = (
            completed_epoch >= self.earliest_stop_epoch
            and self.stale_validations >= self.patience
        )
        return {
            "true_best_improved": true_best_improved,
            "significant_improved": significant_improved,
            "should_stop": should_stop,
            "stale_validations": self.stale_validations,
        }

    def count_stale(self, completed_epoch: int) -> dict:
        """Count an ineligible validation without changing either best."""
        if completed_epoch < 1:
            raise ValueError("completed_epoch must be positive")
        self.stale_validations += 1
        return {
            "true_best_improved": False,
            "significant_improved": False,
            "should_stop": (
                completed_epoch >= self.earliest_stop_epoch
                and self.stale_validations >= self.patience
            ),
            "stale_validations": self.stale_validations,
        }

    def state_dict(self) -> dict:
        return {
            "monitor": self.monitor,
            "direction": "minimize",
            "earliest_stop_epoch": self.earliest_stop_epoch,
            "patience": self.patience,
            "relative_improvement": self.relative_improvement,
            "best_score": self.best_score,
            "significant_best": self.significant_best,
            "stale_validations": self.stale_validations,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "RelativeEarlyStop":
        if state.get("direction", "minimize") != "minimize":
            raise ValueError("only minimize early stopping is supported")
        result = cls(
            monitor=str(state["monitor"]),
            earliest_stop_epoch=int(state["earliest_stop_epoch"]),
            patience=int(state["patience"]),
            relative_improvement=float(state["relative_improvement"]),
            best_score=(
                None if state.get("best_score") is None
                else float(state["best_score"])),
            significant_best=(
                None if state.get("significant_best") is None
                else float(state["significant_best"])),
            stale_validations=int(state.get("stale_validations", 0)),
        )
        for name in ("best_score", "significant_best"):
            value = getattr(result, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"invalid {name}")
        if result.stale_validations < 0:
            raise ValueError("stale_validations cannot be negative")
        return result


@dataclass
class EligibleJADEEarlyStop:
    """Two-phase EBJD monitor required by the frozen training contract."""

    state: RelativeEarlyStop
    eligible_phase: bool = False

    def observe(self, score: float, eligible: bool, completed_epoch: int) -> dict:
        if not self.eligible_phase and eligible:
            self.eligible_phase = True
            self.state.best_score = None
            self.state.significant_best = None
            self.state.stale_validations = 0
            result = self.state.observe(score, completed_epoch)
            result["entered_eligible_phase"] = True
            return result
        if self.eligible_phase and not eligible:
            result = self.state.count_stale(completed_epoch)
            result["entered_eligible_phase"] = False
            return result
        result = self.state.observe(score, completed_epoch)
        result["entered_eligible_phase"] = False
        return result

    def state_dict(self) -> dict:
        return {
            "eligible_phase": self.eligible_phase,
            "relative_state": self.state.state_dict(),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "EligibleJADEEarlyStop":
        return cls(
            state=RelativeEarlyStop.from_state_dict(state["relative_state"]),
            eligible_phase=bool(state.get("eligible_phase", False)),
        )
