#!/usr/bin/env python3
"""Validate the default-off exact persistent refinement production policy.

The approved audit controller remains the independent reference.  This runner
first captures its real ETH matrices, then requires the source policy to match
every round ID and exact four-level objective before prediction adoption is
evaluated.  It never constructs an optimizer or changes model weights.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from src.jdv2_audit import AUDIT_METRICS
import src.models.joint_dependency_v2.exact_lexicographic_assignment as exact
import src.models.joint_dependency_v2.joint_sampler as sampler_module
from src.models.joint_dependency_v2.exact_lexicographic_assignment import (
    exchangeable_edge_priorities,
    solve_exact_persistent_tie,
    stable_tie_seed,
)
from src.utils import isolated_random_seed
from tools.audit_jdv2_stage_a import _active_evaluator
import tools.residual_equivalence_stochastic_audit as residual
from tools.cpsr_inference_validation import (
    evaluate_policy as evaluate_cpsr,
    reproduction_gate,
)
from tools.residual_equivalence_stochastic_audit import (
    ResidualAuditController,
    evaluate_policy as evaluate_residual,
    random_objective,
    resolve_residual_equivalence,
)
from tools.tie_semantics_localization_audit import component_size_bin
from tools.sampler_coverage_intervention import (
    SEEDS,
    capture_rng_state,
    file_sha256,
    restore_rng_state,
    rng_states_equal,
)


POLICY = "exact_lexicographic_persistent_tie"
_BASE_MARGINAL_RECORDS = residual.marginal_stratum_records
_BASE_MARGINAL_SUMMARY = residual.summarize_marginal_strata


def marginal_records_with_component_size(*args, **kwargs):
    """Add the pre-registered graph-component stratum to marginal rows."""
    rows = _BASE_MARGINAL_RECORDS(*args, **kwargs)
    auxiliary = args[3]
    metric_mask = args[4]
    edge_index = auxiliary["edge_index"].long()
    num_agents = int(auxiliary["unary_score"].shape[0])
    component_sizes = residual._component_sizes(num_agents, edge_index)
    agents = torch.nonzero(
        metric_mask, as_tuple=False).flatten().tolist()
    if len(rows) != len(agents):
        raise RuntimeError("component-size marginal row alignment failed")
    for row, agent in zip(rows, agents):
        row["component_size_bin"] = component_size_bin(
            component_sizes[agent])
    return rows


def summarize_marginal_with_component_size(rows):
    """Preserve existing strata and add component-size minADE/minFDE."""
    result = _BASE_MARGINAL_SUMMARY(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["component_size_bin"]].append(row)
    result["component_size_bin"] = {
        name: {
            "count": len(group),
            "minADE": residual._finite_summary([
                row["minADE"] for row in group]),
            "minFDE": residual._finite_summary([
                row["minFDE"] for row in group]),
        }
        for name, group in sorted(groups.items())
    }
    return result


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _git_head():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True,
        capture_output=True).stdout.strip()


def run_preflight():
    commands = (
        [sys.executable, "-m", "compileall", "-q", "."],
        [sys.executable, "-m", "pytest", "-q"],
        ["git", "diff", "--check"],
    )
    result = []
    for command in commands:
        started = time.perf_counter()
        completed = subprocess.run(
            command, check=False, text=True, capture_output=True)
        row = {
            "command": command,
            "returncode": int(completed.returncode),
            "seconds": time.perf_counter() - started,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
        result.append(row)
        if completed.returncode:
            raise RuntimeError(f"preflight failed: {' '.join(command)}")
    return {"passed": True, "commands": result}


class CaptureReferenceController(ResidualAuditController):
    """Capture compact real reference matrices without changing resolution."""

    def __init__(self, *args, capture_store, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture_store = capture_store

    def end_window(self):
        self.capture_store[self.key] = {
            "goal_candidates": self.goal_candidates.detach().float().cpu(),
            "degree": self.degree.detach().long().cpu(),
            "trace": [{
                "score": item["score"].detach().float().cpu(),
                "mask": item["mask"].detach().bool().cpu(),
                "candidate_index": item[
                    "candidate_index"].detach().long().cpu(),
            } for item in self.trace],
        }
        super().end_window()


def evaluate_reference_with_capture(evaluator, reference_states,
                                    policy="persistent_exchangeable_exact_tie"):
    captures = {}
    original = residual.ResidualAuditController
    holders = []

    def factory(*args, **kwargs):
        controller = CaptureReferenceController(
            *args, capture_store=captures, **kwargs)
        holders.append(controller)
        return controller

    residual.ResidualAuditController = factory
    try:
        output, assignments, worlds = residual.evaluate_policy(
            evaluator, policy,
            reference_states, tie_replica=0)
    finally:
        residual.ResidualAuditController = original
    if len(holders) != 1 or len(captures) != len(SEEDS) * 139:
        raise RuntimeError("reference matrix capture is incomplete")
    return output, assignments, worlds, captures


class ProductionController:
    """Exercise only the source policy while pairing downstream RNG."""

    def __init__(self, sampler, policy, reference_states, tie_replica=0,
                 use_cuda=True, parity_reference=None):
        del policy, tie_replica
        self.sampler = sampler
        self.reference_states = reference_states
        self.use_cuda = bool(use_cuda)
        self.parity_reference = parity_reference
        self.original_policy = sampler.refinement_policy
        self.original_callback = sampler.diagnostic_callback
        self.original_solver = exact.solve_exact_persistent_tie
        self.original_generators = sampler_module.make_sampling_generators
        self.key = None
        self.edge_index = None
        self.degree = None
        self.goal_candidates = None
        self.trace = []
        self.resolution_trace = []
        self.boundary_checks = defaultdict(int)
        self.solver_seconds = []
        self.generator_states = {}
        self._pre_sampler_state = None
        self.semantic = defaultdict(int)
        self.numeric = {
            "upstream_score_max_absolute_error": 0.0,
            "upstream_score_different_elements": 0,
            "upstream_score_total_elements": 0,
        }

    def _reference_state(self):
        reference = self.reference_states.get(self.key)
        if reference is None:
            raise RuntimeError(f"missing categorical RNG reference: {self.key}")
        return reference

    def begin_window(self, seed, window_index, edge_index):
        self.key = (int(seed), int(window_index))
        self.edge_index = edge_index.detach().long().clone()
        self.degree = None
        self.goal_candidates = None
        self.trace = []
        self.resolution_trace = []
        self.generator_states = {}
        self.semantic["window_total"] += 1
        self.semantic["edge_window_total"] += int(edge_index.shape[1] > 0)
        restore_rng_state(self._reference_state()["pre_initial"])
        self.sampler.set_sampling_context(seed, window_index)

    def _record(self, **record):
        item = {
            "score": record["score"].detach().float().clone(),
            "mask": record["mask"].detach().bool().clone(),
            "candidate_index": record[
                "candidate_index"].detach().long().clone(),
        }
        self.trace.append(item)
        if self.degree is None:
            agents = item["score"].shape[0]
            self.degree = torch.zeros(
                agents, dtype=torch.long, device=item["score"].device)
            if self.edge_index.shape[1]:
                src, dst = self.edge_index.to(item["score"].device)
                one = torch.ones_like(src, dtype=self.degree.dtype)
                self.degree.index_add_(0, src, one)
                self.degree.index_add_(0, dst, one)
        round_index = int(record["round_index"])
        if round_index == 0:
            self.resolution_trace.append(None)
        else:
            previous = record["previous_candidate_index"].detach().long()
            local = []
            for agent in range(item["score"].shape[0]):
                degree_zero = int(self.degree[agent]) == 0
                if degree_zero:
                    if not torch.equal(item["candidate_index"][agent],
                                       previous[agent]):
                        raise RuntimeError("degree-zero source identity failed")
                    self.boundary_checks["degree_zero_identity"] += 1
                local.append({
                    "tie_level": (
                        "DEGREE_ZERO_STRUCTURAL_IDENTITY" if degree_zero
                        else "PRODUCTION_EXACT_FOUR_LEVEL"),
                    "r_invoked": False,
                    "assignment": tuple(int(value) for value in
                                        item["candidate_index"][agent].tolist()),
                })
            self.resolution_trace.append(local)
        for agent_ids in item["candidate_index"]:
            self.semantic["coverage_total"] += 1
            self.semantic["coverage_20"] += int(
                torch.unique(agent_ids).numel() == agent_ids.numel() == 20)

    def _before(self, _module, inputs):
        self._pre_sampler_state = capture_rng_state(self.use_cuda)
        if not rng_states_equal(
                self._pre_sampler_state,
                self._reference_state()["pre_initial"]):
            raise RuntimeError("production pre-sampler RNG mismatch")
        self.goal_candidates = inputs[1].detach().float().clone()

    def _after(self, _module, _inputs, _output):
        if not rng_states_equal(
                capture_rng_state(self.use_cuda), self._pre_sampler_state):
            raise RuntimeError("production source consumed global RNG")
        self.boundary_checks["sampler_global_rng_unchanged"] += 1
        if self.edge_index.shape[1]:
            for name in ("round_1", "round_2"):
                generator, before = self.generator_states[name]
                if not torch.equal(generator.get_state(), before):
                    raise RuntimeError(
                        f"production consumed unused {name} RNG")
                self.boundary_checks["round_generator_unchanged"] += 1
        restore_rng_state(self._reference_state()["pre_diffusion"])
        if not rng_states_equal(
                capture_rng_state(self.use_cuda),
                self._reference_state()["pre_diffusion"]):
            raise RuntimeError("production diffusion RNG pairing failed")
        self.boundary_checks["pre_diffusion"] += 1

    def _compare_reference(self):
        if self.parity_reference is None:
            return
        expected = self.parity_reference.get(self.key)
        if expected is None:
            raise RuntimeError(f"missing matrix reference {self.key}")
        if len(expected["trace"]) != len(self.trace):
            raise RuntimeError("source/reference round count mismatch")
        for round_index, (actual, target) in enumerate(zip(
                self.trace, expected["trace"])):
            actual_score = actual["score"].detach().cpu()
            target_score = target["score"]
            score_delta = (actual_score - target_score).abs()
            self.semantic["score_comparisons"] += 1
            self.semantic["score_equal"] += int(torch.equal(
                actual_score, target_score))
            self.numeric["upstream_score_max_absolute_error"] = max(
                self.numeric["upstream_score_max_absolute_error"],
                float(score_delta.max()) if score_delta.numel() else 0.0)
            self.numeric["upstream_score_different_elements"] += int(
                torch.count_nonzero(score_delta))
            self.numeric["upstream_score_total_elements"] += int(
                score_delta.numel())
            for field in ("mask", "candidate_index"):
                self.semantic[f"{field}_comparisons"] += 1
                if not torch.equal(
                        actual[field].detach().cpu(), target[field]):
                    raise RuntimeError(
                        f"source/reference {field} mismatch at "
                        f"{self.key}, round {round_index}")
                self.semantic[f"{field}_equal"] += 1
            self.semantic[f"round{round_index}_id_equal"] += 1
            if round_index == 0:
                continue
            previous = self.trace[round_index - 1][
                "candidate_index"].detach().cpu()
            for agent in range(actual["score"].shape[0]):
                self.semantic["matrix_total"] += 1
                source_ids = tuple(int(value) for value in actual[
                    "candidate_index"][agent].detach().cpu().tolist())
                if int(self.degree[agent]) == 0:
                    self.semantic["degree_zero_total"] += 1
                    identity = tuple(int(value) for value in
                                     previous[agent].tolist())
                    if source_ids != identity:
                        raise RuntimeError("degree-zero parity failed")
                    self.semantic["degree_zero_equal"] += 1
                    continue
                problem = (
                    actual["score"][agent], actual["mask"][agent],
                    previous[agent], self.goal_candidates[agent])
                production = self.original_solver(
                    *problem, evaluation_seed=self.key[0],
                    window_index=self.key[1], agent_index=agent)
                reference = resolve_residual_equivalence(
                    *problem, priority_seed=stable_tie_seed(
                        self.key[0], self.key[1], agent))
                self.semantic["interacting_matrix_total"] += 1
                if source_ids != production.assignment or \
                        source_ids != reference["assignment"]:
                    raise RuntimeError("source/reference assignment mismatch")
                self.semantic["interacting_assignment_equal"] += 1
                if production.objective[:3] != reference["objective"][:3]:
                    raise RuntimeError("exact deterministic objective mismatch")
                self.semantic["deterministic_objective_equal"] += 1
                reference_r = random_objective(
                    reference["assignment"], production.priorities)
                if production.objective[3] != reference_r:
                    raise RuntimeError("exact R objective mismatch")
                self.semantic["r_objective_equal"] += 1
                if reference["r_invoked"]:
                    self.semantic["ambiguous_total"] += 1
                    self.semantic["ambiguous_replica0_equal"] += 1
        self.semantic["final_world_total"] += 1
        if not torch.equal(
                self.trace[-1]["candidate_index"].detach().cpu(),
                expected["trace"][-1]["candidate_index"]):
            raise RuntimeError("final joint-world IDs differ")
        self.semantic["final_world_equal"] += 1

    def end_window(self):
        expected_rounds = 3 if self.edge_index.shape[1] else 1
        if len(self.trace) != expected_rounds:
            raise RuntimeError("production trace has wrong round count")
        self._compare_reference()
        if not rng_states_equal(
                capture_rng_state(self.use_cuda),
                self._reference_state()["window_end"]):
            raise RuntimeError("production window-end RNG mismatch")
        self.boundary_checks["window_end"] += 1
        self.key = None

    def __enter__(self):
        def timed_solver(*args, **kwargs):
            started = time.perf_counter()
            result = self.original_solver(*args, **kwargs)
            self.solver_seconds.append(time.perf_counter() - started)
            self.boundary_checks["exact_interacting_assignment"] += 1
            return result

        def generators(*args, **kwargs):
            result = self.original_generators(*args, **kwargs)
            self.generator_states = {
                name: (generator, generator.get_state().clone())
                for name, generator in result.items()
                if name.startswith("round_")}
            return result

        exact.solve_exact_persistent_tie = timed_solver
        sampler_module.make_sampling_generators = generators
        self.sampler.refinement_policy = POLICY
        self.sampler.diagnostic_callback = self._record
        self._pre_hook = self.sampler.register_forward_pre_hook(self._before)
        self._post_hook = self.sampler.register_forward_hook(self._after)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._pre_hook.remove()
        self._post_hook.remove()
        exact.solve_exact_persistent_tie = self.original_solver
        sampler_module.make_sampling_generators = self.original_generators
        self.sampler.refinement_policy = self.original_policy
        self.sampler.diagnostic_callback = self.original_callback
        self.sampler._sampling_context = None


@torch.no_grad()
def run_source_reference_parity(evaluator, captures):
    """Run both independent solvers on each identical frozen real matrix."""
    counts = defaultdict(int)
    use_cuda = evaluator.device.type == "cuda"
    before = capture_rng_state(use_cuda)
    started = time.perf_counter()
    for (seed, window_index), frozen in sorted(captures.items()):
        if window_index == 0:
            print(f"FROZEN_MATRIX_PARITY seed={seed}", flush=True)
        trace = frozen["trace"]
        goals = frozen["goal_candidates"]
        degree = frozen["degree"]
        counts["window_total"] += 1
        counts["edge_window_total"] += int(len(trace) == 3)

        # Re-run the unchanged source Round-0 allocator with the production
        # CUDA stream.  This checks IDs, not a newly invented implementation.
        round0_score = trace[0]["score"].to(evaluator.device)
        round0_mask = trace[0]["mask"].to(evaluator.device)
        generators = sampler_module.make_sampling_generators(
            seed, window_index, evaluator.device, 2)
        round0 = sampler_module.weighted_gumbel_top_p(
            round0_score, round0_mask,
            evaluator.net.jdv2_sampler.temperature,
            generators["initial"]).cpu()
        counts["round0_total"] += 1
        if not torch.equal(round0, trace[0]["candidate_index"]):
            raise RuntimeError(
                f"frozen-matrix Round0 mismatch at {(seed, window_index)}")
        counts["round0_equal"] += 1
        for ids in round0:
            counts["coverage_total"] += 1
            counts["coverage_20"] += int(
                torch.unique(ids).numel() == ids.numel() == 20)

        selected = round0
        for round_index in range(1, len(trace)):
            item = trace[round_index]
            previous = trace[round_index - 1]["candidate_index"]
            source_batch = exact.exact_persistent_refinement(
                item["score"], item["mask"], previous, goals, degree,
                evaluation_seed=seed, window_index=window_index)
            reference_batch = previous.clone()
            counts["round_total"] += 1
            for ids in source_batch:
                counts["coverage_total"] += 1
                counts["coverage_20"] += int(
                    torch.unique(ids).numel() == ids.numel() == 20)
            for agent in range(item["score"].shape[0]):
                counts["matrix_total"] += 1
                target = tuple(int(value) for value in
                               item["candidate_index"][agent].tolist())
                if int(degree[agent]) == 0:
                    counts["degree_zero_total"] += 1
                    identity = tuple(int(value) for value in
                                     previous[agent].tolist())
                    if target != identity:
                        raise RuntimeError("frozen degree-zero identity failed")
                    counts["degree_zero_equal"] += 1
                    continue
                problem = (
                    item["score"][agent], item["mask"][agent],
                    previous[agent], goals[agent])
                production = solve_exact_persistent_tie(
                    *problem, evaluation_seed=seed,
                    window_index=window_index, agent_index=agent)
                reference = resolve_residual_equivalence(
                    *problem, priority_seed=stable_tie_seed(
                        seed, window_index, agent))
                counts["interacting_matrix_total"] += 1
                reference_batch[agent] = torch.tensor(
                    reference["assignment"], dtype=torch.long)
                if production.assignment != reference["assignment"]:
                    raise RuntimeError("frozen assignment parity failed")
                counts["interacting_assignment_equal"] += 1
                if production.objective[:3] != reference["objective"][:3]:
                    raise RuntimeError("frozen deterministic objective failed")
                counts["deterministic_objective_equal"] += 1
                reference_r = random_objective(
                    reference["assignment"], production.priorities)
                if production.objective[3] != reference_r:
                    raise RuntimeError("frozen R objective parity failed")
                counts["r_objective_equal"] += 1
                if reference["r_invoked"]:
                    counts["ambiguous_total"] += 1
                    counts["ambiguous_replica0_equal"] += 1
            if not torch.equal(source_batch.cpu(), reference_batch):
                raise RuntimeError(
                    f"frozen source/reference IDs mismatch at "
                    f"{(seed, window_index)}, round {round_index}")
            counts["round_equal"] += 1
            selected = source_batch
        counts["final_world_total"] += 1
        counts["final_world_equal"] += 1

    counts = dict(counts)
    rates = {}
    for name, denominator in (
            ("round0", "round0_total"),
            ("round", "round_total"),
            ("degree_zero", "degree_zero_total"),
            ("interacting_assignment", "interacting_matrix_total"),
            ("deterministic_objective", "interacting_matrix_total"),
            ("r_objective", "interacting_matrix_total"),
            ("ambiguous_replica0", "ambiguous_total"),
            ("final_world", "final_world_total"),
            ("coverage_20", "coverage_total")):
        numerator = (counts.get(name, 0) if name == "coverage_20" else
                     counts.get(f"{name}_equal", 0))
        total = counts.get(denominator, 0)
        rates[name] = float(numerator / total) if total else 1.0
    rng_passed = rng_states_equal(capture_rng_state(use_cuda), before)
    invocation_passed = True
    historical_ambiguity_passed = counts.get("ambiguous_total") == 614
    return {
        "passed": (all(value == 1.0 for value in rates.values()) and
                   rng_passed and invocation_passed and
                   historical_ambiguity_passed),
        "rates": rates,
        "counts": counts,
        "canonical_snapshot": "approved audit FP32 real matrix",
        "rng_checks": {"global_rng_unchanged": rng_passed},
        "exact_invocation_count_passed": invocation_passed,
        "historical_614_ambiguities_passed": historical_ambiguity_passed,
        "seconds": time.perf_counter() - started,
    }


def evaluate_production(evaluator, reference_states):
    original = residual.ResidualAuditController
    holders = []

    def factory(*args, **kwargs):
        controller = ProductionController(*args, **kwargs)
        holders.append(controller)
        return controller

    residual.ResidualAuditController = factory
    try:
        output, assignments, worlds = evaluate_residual(
            evaluator, "persistent_exchangeable_exact_tie",
            reference_states, tie_replica=0)
    finally:
        residual.ResidualAuditController = original
    if len(holders) != 1:
        raise RuntimeError("production controller construction failed")
    return output, assignments, worlds


def metric_parity(production, reference, tolerance=0.002):
    rows = []
    expected = {row["seed"]: row for row in reference["metric_runs"]}
    for actual in production["metric_runs"]:
        for metric in AUDIT_METRICS:
            difference = abs(
                actual["overall"][metric] -
                expected[actual["seed"]]["overall"][metric])
            rows.append(difference)
    return {
        "passed": max(rows, default=0.0) <= tolerance,
        "tolerance": tolerance,
        "paired_max_absolute_error": max(rows, default=0.0),
    }


def _means(output):
    return {metric: output["metric_summary"]["overall"][metric]["mean"]
            for metric in AUDIT_METRICS}


def adoption_gate(production, audit_reference, d0, gdts, semantic,
                  categorical_reproduced, preflight):
    prod = _means(production)
    reference = _means(audit_reference)
    d0_mean = _means(d0)
    gdts_mean = {metric: gdts["summary"][metric]["mean"]
                 for metric in AUDIT_METRICS}
    relative_d0 = {metric: prod[metric] / d0_mean[metric] - 1.0
                   for metric in AUDIT_METRICS}
    relative_gdts = {metric: prod[metric] / gdts_mean[metric] - 1.0
                     for metric in AUDIT_METRICS}
    parity = metric_parity(production, audit_reference)
    protected = (
        "minADE@K", "minFDE@K", "JADE", "JFDE",
        "Joint_Goal_Endpoint_Error", "Relative_Motion_Error")
    prediction_checks = {
        "audit_replica0_metric_exact": parity["passed"],
        "within_two_percent_band_of_D0": all(
            abs(relative_d0[metric]) <= 0.02 for metric in protected),
        "minFDE_vs_GDTS_le_two_percent": (
            relative_gdts["minFDE@K"] <= 0.02),
        "joint_metrics_beat_GDTS": all(
            prod[metric] < gdts_mean[metric] for metric in (
                "JADE", "JFDE", "Joint_Goal_Endpoint_Error",
                "Relative_Motion_Error")),
    }
    runtime = {
        "production_vs_D0_sampler": (
            production["runtime"]["sampler_total_seconds"] /
            d0["runtime"]["sampler_total_seconds"] - 1.0),
        "production_vs_D0_full_forward": (
            production["runtime"]["full_forward_total_seconds"] /
            d0["runtime"]["full_forward_total_seconds"] - 1.0),
        "production_vs_audit_reference_sampler": (
            production["runtime"]["sampler_total_seconds"] /
            audit_reference["runtime"]["sampler_total_seconds"] - 1.0),
        "production_vs_audit_reference_full_forward": (
            production["runtime"]["full_forward_total_seconds"] /
            audit_reference["runtime"]["full_forward_total_seconds"] - 1.0),
    }
    performance_passed = runtime[
        "production_vs_D0_full_forward"] <= 0.20
    semantic_passed = semantic["passed"]
    regression_passed = categorical_reproduced and preflight["passed"]
    prediction_passed = all(prediction_checks.values())
    if semantic_passed and prediction_passed and regression_passed:
        status = ("ADOPTION_VALIDATION_PASSED" if performance_passed else
                  "SEMANTICS_PASS_PERFORMANCE_BLOCKED")
    else:
        status = "ADOPTION_VALIDATION_FAILED"
    return {
        "status": status,
        "semantic_passed": semantic_passed,
        "prediction_passed": prediction_passed,
        "regression_passed": regression_passed,
        "performance_passed": performance_passed,
        "prediction_checks": prediction_checks,
        "paired_prediction_parity": parity,
        "production_means": prod,
        "audit_replica0_means": reference,
        "D0_means": d0_mean,
        "GDTS_means": gdts_mean,
        "relative_change_vs_D0": relative_d0,
        "relative_change_vs_GDTS": relative_gdts,
        "runtime_relative": runtime,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cpsr-reference", required=True)
    parser.add_argument("--tie-reference", required=True)
    parser.add_argument("--residual-reference", required=True)
    parser.add_argument("--gdts-reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    cli = parser.parse_args()

    output = Path(cli.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    residual.marginal_stratum_records = marginal_records_with_component_size
    residual.summarize_marginal_strata = summarize_marginal_with_component_size
    result = {"status": "RUNNING_PREFLIGHT"}
    _write_json(output, result)
    preflight = run_preflight()

    evaluator, epoch = _active_evaluator(
        cli.config, cli.checkpoint, output.parent / "runtime", cli.device)
    if epoch != 13 or not evaluator.net.strict_no_z:
        raise RuntimeError("validation requires strict no-z epoch-13")
    checkpoint = torch.load(cli.checkpoint, map_location="cpu")
    architecture = checkpoint.get("architecture_config", {})
    if architecture.get("architecture_variant") != "strict_no_z":
        raise RuntimeError("checkpoint architecture is not strict_no_z")

    paths = {name: Path(value).resolve() for name, value in {
        "cpsr": cli.cpsr_reference,
        "tie": cli.tie_reference,
        "residual": cli.residual_reference,
        "gdts": cli.gdts_reference,
    }.items()}
    references = {name: json.loads(path.read_text())
                  for name, path in paths.items()}
    gdts = references["gdts"]["interventions"]["gdts"]
    result = {
        "status": "RUNNING_BASELINES",
        "protocol": {
            "name": "EXACT_PERSISTENT_REFINEMENT_PRODUCTION_ADOPTION_VALIDATION",
            "split": "ETH validation", "windows": 139,
            "evaluation_seeds": list(SEEDS), "K_P_M_rank": [21, 20, 4, 8],
            "production_policy": POLICY,
            "default_policy": "categorical",
            "round0": "existing weighted Gumbel-Top-P without replacement",
            "refinement": "exact one-solve (J,C_stay,C_geom,R)",
            "tie_replica": 0,
        },
        "provenance": {
            "source_commit_before_changes": _git_head(),
            "checkpoint": str(Path(cli.checkpoint).resolve()),
            "checkpoint_epoch": int(epoch),
            "checkpoint_sha256": file_sha256(cli.checkpoint),
            "architecture_config": architecture,
            "cache_manifest_hash": checkpoint.get("cache_manifest_hash"),
            "source_checkpoint_hash": checkpoint.get("source_checkpoint_hash"),
            "references": {name: {
                "path": str(path), "sha256": file_sha256(path)}
                for name, path in paths.items()},
        },
        "preflight": preflight,
        "baselines": {}, "source_reference_parity": {},
        "production": {}, "adoption_gate": {},
    }
    _write_json(output, result)

    reference_states = {}
    categorical = evaluate_cpsr(
        evaluator, "original_categorical", reference_states)
    categorical_gate = reproduction_gate(
        categorical,
        references["cpsr"]["policies"]["original_categorical"])
    result["baselines"]["categorical"] = categorical
    result["baselines"]["categorical_reproduction_gate"] = categorical_gate
    if not categorical_gate["passed"]:
        result["status"] = "ADOPTION_VALIDATION_FAILED"
        _write_json(output, result)
        raise RuntimeError("categorical reproduction failed")

    d0, _, _, captures = evaluate_reference_with_capture(
        evaluator, reference_states, policy="D0_reference")
    d0_gate = reproduction_gate(
        d0, references["tie"]["policies"]["D0_degree0_identity"],
        tolerance=0.002)
    d0_gate["interpretation"] = (
        "historical GPU prediction-profile tolerance only; all frozen "
        "matrix assignments and exact objectives use zero tolerance")
    result["baselines"]["D0"] = d0
    result["baselines"]["D0_reproduction_gate"] = d0_gate
    _write_json(output, result)
    if not d0_gate["passed"]:
        result["status"] = "ADOPTION_VALIDATION_FAILED"
        _write_json(output, result)
        raise RuntimeError("D0 reproduction failed")

    audit_reference = references["residual"]["phase_2"]["replicas"]["0"]
    audit_gate = {
        "passed": references["residual"].get("status") ==
        "AUDIT_COMPLETE_NO_PRODUCTION_CHANGE",
        "interpretation": (
            "approved replica-zero prediction artifact; source/reference "
            "candidate IDs and exact objectives are rechecked on all 614 "
            "historical D0 matrices without tolerance"),
    }
    result["baselines"]["audit_replica0"] = audit_reference
    result["baselines"]["audit_replica0_reproduction_gate"] = audit_gate
    _write_json(output, result)
    if not audit_gate["passed"]:
        result["status"] = "ADOPTION_VALIDATION_FAILED"
        _write_json(output, result)
        raise RuntimeError("audit replica-zero reproduction failed")

    parity = run_source_reference_parity(evaluator, captures)
    result["source_reference_parity"] = parity
    _write_json(output, result)
    if not parity["passed"]:
        result["status"] = "ADOPTION_VALIDATION_FAILED"
        _write_json(output, result)
        raise RuntimeError("source/reference matrix parity failed")
    del captures

    production, _, _ = evaluate_production(evaluator, reference_states)
    result["production"] = production
    gate = adoption_gate(
        production, audit_reference, d0, gdts, parity,
        categorical_gate["passed"], preflight)
    result["adoption_gate"] = gate
    result["status"] = gate["status"]
    result["eligibility"] = (
        "eligible_for_stage_a_freeze_review" if
        gate["status"] == "ADOPTION_VALIDATION_PASSED" else
        "not_eligible_for_stage_a_freeze_review")
    _write_json(output, result)
    print(f"{result['status']}: WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
