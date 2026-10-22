"""Marginal-constrained validation checkpoint selection."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConstrainedSelector:
    baseline_minade: float
    baseline_minfde: float
    tolerance: float = 0.0
    candidates: list[dict] = field(default_factory=list)
    best_eligible: dict | None = None
    least_violation: dict | None = None

    @staticmethod
    def _dominates(left: dict, right: dict) -> bool:
        names = ("minADE", "minFDE", "JADE", "JFDE")
        left_values = [float(left["metrics"][name]) for name in names]
        right_values = [float(right["metrics"][name]) for name in names]
        return all(a <= b for a, b in zip(left_values, right_values, strict=True)) and any(
            a < b for a, b in zip(left_values, right_values, strict=True))

    def pareto_candidates(self) -> list[dict]:
        return [candidate for candidate in self.candidates if not any(
            other is not candidate and self._dominates(other, candidate)
            for other in self.candidates)]

    def consider(self, epoch: int, metrics: dict) -> dict:
        ade_violation = max(0.0, float(metrics["minADE"]) - self.baseline_minade - self.tolerance)
        fde_violation = max(0.0, float(metrics["minFDE"]) - self.baseline_minfde - self.tolerance)
        eligible = ade_violation == 0 and fde_violation == 0
        candidate = {
            "epoch": int(epoch), "metrics": metrics,
            "marginal_violations": {"minADE": ade_violation, "minFDE": fde_violation},
            "total_violation": ade_violation + fde_violation,
            "joint_score": float(metrics["JADE"]) + 0.5 * float(metrics["JFDE"]),
            "simultaneous_improvement": eligible,
        }
        self.candidates.append(candidate)
        if eligible:
            key = (candidate["joint_score"], candidate["epoch"])
            if self.best_eligible is None or key < (
                self.best_eligible["joint_score"], self.best_eligible["epoch"]):
                self.best_eligible = candidate
        violation_key = (candidate["total_violation"], candidate["joint_score"], candidate["epoch"])
        if self.least_violation is None or violation_key < (
            self.least_violation["total_violation"],
            self.least_violation["joint_score"], self.least_violation["epoch"]):
            self.least_violation = candidate
        return candidate

    def state_dict(self) -> dict:
        return {
            "baseline_minade": self.baseline_minade,
            "baseline_minfde": self.baseline_minfde,
            "tolerance": self.tolerance,
            "candidates": self.candidates,
            "best_eligible": self.best_eligible,
            "least_violation": self.least_violation,
            "pareto_candidates": self.pareto_candidates(),
            "simultaneous_improvement": self.best_eligible is not None,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "ConstrainedSelector":
        selector = cls(
            float(state["baseline_minade"]), float(state["baseline_minfde"]),
            float(state.get("tolerance", 0)))
        selector.candidates = list(state.get("candidates", []))
        selector.best_eligible = state.get("best_eligible")
        selector.least_violation = state.get("least_violation")
        return selector
