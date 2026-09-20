#!/usr/bin/env python3
"""Inference-only coverage interventions for the strict-no-z sampler.

The default model/sampler path is never changed.  This audit-only runner
temporarily intercepts the sampler's categorical primitive, changes only its
first call in each window, and pairs all downstream RNG states to an exact
unmodified baseline run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from src.jdv2_audit import AUDIT_METRICS
from src.metrics import compute_metric_mask
import src.models.joint_dependency_v2.joint_sampler as sampler_module
from src.utils import isolated_random_seed
from tools.audit_jdv2_stage_a import _active_evaluator


POLICIES = (
    "iid_with_replacement",
    "weighted_without_replacement",
    "deterministic_top_p",
)
SEEDS = (2035, 2036, 2037, 2038, 2039)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_numpy_state(state):
    return (state[0], state[1].copy(), state[2], state[3], state[4])


@dataclass
class RNGSnapshot:
    """Complete process RNG state used by the validation protocol."""

    python: object
    numpy: tuple
    torch_cpu: torch.Tensor
    torch_cuda: Optional[list]
    cudnn_deterministic: bool
    cudnn_benchmark: bool


def capture_rng_state(use_cuda=True):
    cuda = bool(use_cuda and torch.cuda.is_available())
    return RNGSnapshot(
        python=copy.deepcopy(random.getstate()),
        numpy=_copy_numpy_state(np.random.get_state()),
        torch_cpu=torch.get_rng_state().clone(),
        torch_cuda=(
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if cuda else None),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
    )


def restore_rng_state(state: RNGSnapshot):
    random.setstate(copy.deepcopy(state.python))
    np.random.set_state(_copy_numpy_state(state.numpy))
    torch.set_rng_state(state.torch_cpu.clone())
    if state.torch_cuda is not None:
        torch.cuda.set_rng_state_all(
            [value.clone() for value in state.torch_cuda])
    torch.backends.cudnn.deterministic = state.cudnn_deterministic
    torch.backends.cudnn.benchmark = state.cudnn_benchmark


def rng_states_equal(left: RNGSnapshot, right: RNGSnapshot):
    numpy_equal = (
        left.numpy[0] == right.numpy[0] and
        np.array_equal(left.numpy[1], right.numpy[1]) and
        left.numpy[2:] == right.numpy[2:])
    if left.torch_cuda is None or right.torch_cuda is None:
        cuda_equal = left.torch_cuda is None and right.torch_cuda is None
    else:
        cuda_equal = len(left.torch_cuda) == len(right.torch_cuda) and all(
            torch.equal(a, b) for a, b in zip(
                left.torch_cuda, right.torch_cuda))
    return (
        left.python == right.python and numpy_equal and
        torch.equal(left.torch_cpu, right.torch_cpu) and cuda_equal and
        left.cudnn_deterministic == right.cudnn_deterministic and
        left.cudnn_benchmark == right.cudnn_benchmark)


def _validate_initial_inputs(score, mask, num_samples):
    if score.ndim != 3 or mask.shape != score.shape:
        raise ValueError("initial score/mask must have shape [N,P,K]")
    if score.shape[1] != num_samples:
        raise ValueError("initial score P axis differs from requested P")
    if num_samples > score.shape[-1]:
        raise ValueError(
            f"without-replacement allocation requires P<=K; "
            f"got P={num_samples}, K={score.shape[-1]}")
    if not torch.isfinite(score[mask]).all():
        raise FloatingPointError("valid initial logits contain NaN/Inf")
    if not torch.equal(score, score[:, :1].expand_as(score)):
        raise ValueError("initial slot logits must be identical across P")
    if not torch.equal(mask, mask[:, :1].expand_as(mask)):
        raise ValueError("initial candidate mask must be identical across P")
    local_mask = mask[:, 0].bool()
    if bool((local_mask.sum(dim=-1) < num_samples).any()):
        raise ValueError("fewer than P valid candidates; refusing fallback")
    logits = score[:, 0].float()
    return logits, local_mask


def initial_candidate_ids(score, mask, temperature, sampling_mode, generator,
                          policy, num_samples=None):
    """Return initial IDs, altering no score and never silently falling back."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    if policy == "iid_with_replacement":
        return sampler_module._categorical(
            score, mask, temperature, sampling_mode, generator)
    if sampling_mode != "sample":
        raise ValueError("coverage interventions require deployed sample mode")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    num_samples = int(score.shape[1] if num_samples is None else num_samples)
    logits, local_mask = _validate_initial_inputs(
        score, mask, num_samples)
    weighted_logits = (logits / float(temperature)).masked_fill(
        ~local_mask, float("-inf"))
    if policy == "weighted_without_replacement":
        uniform = torch.rand(
            weighted_logits.shape, dtype=torch.float32,
            device=weighted_logits.device, generator=generator)
        finfo = torch.finfo(uniform.dtype)
        uniform = uniform.clamp(min=finfo.tiny, max=1.0 - finfo.eps)
        gumbel = -torch.log(-torch.log(uniform))
        key = weighted_logits + gumbel
    else:
        key = weighted_logits
    selected = torch.topk(
        key, k=num_samples, dim=-1, largest=True, sorted=True).indices
    if selected.shape != (score.shape[0], num_samples):
        raise RuntimeError("intervention returned an invalid shape")
    if bool((selected < 0).any()) or bool((selected >= score.shape[-1]).any()):
        raise RuntimeError("intervention returned an invalid candidate ID")
    if not local_mask.gather(1, selected).all():
        raise RuntimeError("intervention selected a masked candidate")
    sorted_ids = selected.sort(dim=-1).values
    if num_samples > 1 and bool((
            sorted_ids[:, 1:] == sorted_ids[:, :-1]).any()):
        raise RuntimeError("without-replacement intervention produced duplicates")
    return selected


def _initial_generator(seed, window_index, device):
    # Independent audit-only stream for Gumbel allocation.  A stable integer
    # formula avoids Python's process-randomized hash().
    value = (int(seed) * 1_000_003 + int(window_index) * 97_409 + 17) % (
        2 ** 63 - 1)
    generator = torch.Generator(device=device)
    generator.manual_seed(value)
    return generator


class PairedSamplerIntervention:
    """Intercept only initial allocation and pair downstream RNG boundaries."""

    def __init__(self, policy, reference_states, use_cuda=True):
        if policy not in POLICIES:
            raise ValueError(f"unknown policy: {policy}")
        self.policy = policy
        self.reference_states = reference_states
        self.use_cuda = bool(use_cuda)
        self.original = sampler_module._categorical
        self.key = None
        self.call_index = 0
        self.expected_calls = None
        self.init_generator = None
        self.boundary_checks = defaultdict(int)

    def begin_window(self, seed, window_index, edge_count, device):
        self.key = (int(seed), int(window_index))
        self.call_index = 0
        self.expected_calls = 3 if int(edge_count) > 0 else 1
        self.init_generator = (
            _initial_generator(seed, window_index, device)
            if self.policy == "weighted_without_replacement" else None)

    def end_window(self):
        if self.call_index != self.expected_calls:
            raise RuntimeError(
                f"unexpected categorical call count: {self.call_index} != "
                f"{self.expected_calls}")
        state = capture_rng_state(self.use_cuda)
        if self.policy == "iid_with_replacement":
            self.reference_states[self.key]["window_end"] = state
        elif not rng_states_equal(
                state, self.reference_states[self.key]["window_end"]):
            raise RuntimeError("downstream RNG diverged after model forward")
        else:
            self.boundary_checks["window_end"] += 1
        self.key = None

    def __enter__(self):
        if sampler_module._categorical is not self.original:
            raise RuntimeError("sampler categorical primitive already patched")

        def categorical(score, mask, temperature, sampling_mode, generator):
            if self.key is None:
                raise RuntimeError("begin_window was not called")
            index = self.call_index
            if index >= self.expected_calls:
                raise RuntimeError("too many categorical calls in one window")
            if index == 0:
                if self.policy == "iid_with_replacement":
                    self.reference_states[self.key] = {
                        "pre_initial": capture_rng_state(self.use_cuda)}
                    result = self.original(
                        score, mask, temperature, sampling_mode, generator)
                    self.reference_states[self.key]["pre_refinement"] = \
                        capture_rng_state(self.use_cuda)
                else:
                    reference = self.reference_states.get(self.key)
                    if reference is None:
                        raise RuntimeError("missing Policy-A RNG reference")
                    restore_rng_state(reference["pre_initial"])
                    result = initial_candidate_ids(
                        score, mask, temperature, sampling_mode,
                        self.init_generator, self.policy,
                        num_samples=score.shape[1])
                    restore_rng_state(reference["pre_refinement"])
                    if not rng_states_equal(
                            capture_rng_state(self.use_cuda),
                            reference["pre_refinement"]):
                        raise RuntimeError("refinement RNG restore failed")
                    self.boundary_checks["pre_refinement"] += 1
            else:
                result = self.original(
                    score, mask, temperature, sampling_mode, generator)

            if index == self.expected_calls - 1:
                if self.policy == "iid_with_replacement":
                    self.reference_states[self.key]["pre_diffusion"] = \
                        capture_rng_state(self.use_cuda)
                else:
                    reference = self.reference_states[self.key]
                    # The normal refinement draws have the same shapes, but
                    # restore explicitly so downstream pairing is guaranteed.
                    restore_rng_state(reference["pre_diffusion"])
                    if not rng_states_equal(
                            capture_rng_state(self.use_cuda),
                            reference["pre_diffusion"]):
                        raise RuntimeError("diffusion RNG restore failed")
                    self.boundary_checks["pre_diffusion"] += 1
            self.call_index += 1
            return result

        sampler_module._categorical = categorical
        return self

    def __exit__(self, exc_type, exc, traceback):
        sampler_module._categorical = self.original


def _rank_tensor(score):
    order = torch.argsort(score.float(), dim=-1, descending=True, stable=True)
    rank = torch.empty_like(order)
    position = torch.arange(
        1, score.shape[-1] + 1, device=score.device)[None].expand_as(order)
    rank.scatter_(1, order, position)
    return rank


def _world_ground_truth(net, inputs):
    pixel = inputs["x_augmented"][:, :, 6:8].detach().float() * \
        float(net.args.down_factor)
    return inputs["scene"].make_world_coord_torch(pixel)


def _world_predictions(net, predictions, inputs):
    pixel = predictions.detach().float() * float(net.args.down_factor)
    return torch.stack([
        inputs["scene"].make_world_coord_torch(sample) for sample in pixel])


def _degrees(num_agents, edge_index):
    degree = torch.zeros(
        num_agents, dtype=torch.long, device=edge_index.device)
    if edge_index.shape[1]:
        one = torch.ones(
            edge_index.shape[1], dtype=torch.long, device=edge_index.device)
        degree.index_add_(0, edge_index[0].long(), one)
        degree.index_add_(0, edge_index[1].long(), one)
    return degree


def _agent_count_bin(count):
    if count == 1:
        return "N=1"
    if count == 2:
        return "N=2"
    if count <= 4:
        return "N=3-4"
    if count <= 8:
        return "N=5-8"
    return "N>=9"


def _degree_bin(value):
    if value == 0:
        return "degree=0"
    if value == 1:
        return "degree=1"
    if value <= 3:
        return "degree=2-3"
    return "degree>=4"


def _agent_records(net, inputs, output, auxiliary, metric_mask, policy, seed,
                   window_index):
    candidate = auxiliary["goal_candidates_world"].float()
    bank_score = auxiliary["candidate_log_prior"].float()
    unary_score = auxiliary["unary_score"].float()
    initial = auxiliary["sampled"]["initial_candidate_index"].long()
    final = auxiliary["joint_candidate_index"][
        :, :net.args.num_samples].long()
    if initial.shape != final.shape:
        raise RuntimeError("initial/final candidate-index shape mismatch")
    num_agents, num_samples = initial.shape
    gt_goal = _world_ground_truth(net, inputs)[-1]
    candidate_error = torch.linalg.vector_norm(
        candidate - gt_goal[:, None], dim=-1)
    bank_rank = _rank_tensor(bank_score)
    unary_rank = _rank_tensor(unary_score)
    bank_oracle, oracle_id = candidate_error.min(dim=1)
    oracle_bank_rank = bank_rank.gather(1, oracle_id[:, None]).squeeze(1)
    oracle_unary_rank = unary_rank.gather(1, oracle_id[:, None]).squeeze(1)
    initial_error = candidate_error.gather(1, initial)
    final_error = candidate_error.gather(1, final)
    initial_bank_rank = bank_rank.gather(1, initial)
    final_bank_rank = bank_rank.gather(1, final)
    initial_unary_rank = unary_rank.gather(1, initial)
    final_unary_rank = unary_rank.gather(1, final)
    prediction_world = _world_predictions(net, output, inputs)
    trajectory_error = torch.linalg.vector_norm(
        prediction_world[:, -1] - gt_goal[None], dim=-1).transpose(0, 1)
    trajectory_min, best_slot = trajectory_error.min(dim=1)
    best_slot_goal = final_error.gather(
        1, best_slot[:, None]).squeeze(1)
    edge_index = auxiliary["edge_index"].long()
    edge_class = "E>0" if edge_index.shape[1] else "E=0"
    degree = _degrees(num_agents, edge_index)
    rows = []
    for agent in torch.nonzero(metric_mask, as_tuple=False).flatten().tolist():
        initial_unique = int(torch.unique(initial[agent]).numel())
        final_unique = int(torch.unique(final[agent]).numel())
        local_degree = int(degree[agent].item())
        row = {
            "policy": policy, "seed": int(seed),
            "window": int(window_index), "agent": int(agent),
            "edge_class": edge_class, "agent_count": int(num_agents),
            "agent_count_bin": _agent_count_bin(num_agents),
            "degree": local_degree, "degree_bin": _degree_bin(local_degree),
            "num_slots": int(num_samples),
            "initial_unique_count": initial_unique,
            "initial_duplicate_count": int(num_samples - initial_unique),
            "initial_unique_ratio": initial_unique / float(num_samples),
            "final_unique_count": final_unique,
            "final_duplicate_count": int(num_samples - final_unique),
            "candidate_changed_fraction": float((
                initial[agent] != final[agent]).float().mean().cpu()),
            "candidate_bank_oracle_error": float(bank_oracle[agent].cpu()),
            "initial_slot_oracle_error": float(
                initial_error[agent].min().cpu()),
            "final_selected_goal_oracle_error": float(
                final_error[agent].min().cpu()),
            "trajectory_minFDE": float(trajectory_min[agent].cpu()),
            "trajectory_best_slot_goal_error": float(
                best_slot_goal[agent].cpu()),
            "gt_oracle_frozen_rank": int(oracle_bank_rank[agent].cpu()),
            "gt_oracle_unary_rank": int(oracle_unary_rank[agent].cpu()),
            "mean_initial_frozen_rank": float(
                initial_bank_rank[agent].float().mean().cpu()),
            "mean_final_frozen_rank": float(
                final_bank_rank[agent].float().mean().cpu()),
            "mean_initial_unary_rank": float(
                initial_unary_rank[agent].float().mean().cpu()),
            "mean_final_unary_rank": float(
                final_unary_rank[agent].float().mean().cpu()),
            "refinement_frozen_rank_delta": float((
                final_bank_rank[agent].float() -
                initial_bank_rank[agent].float()).mean().cpu()),
            "refinement_unary_rank_delta": float((
                final_unary_rank[agent].float() -
                initial_unary_rank[agent].float()).mean().cpu()),
        }
        for family, initial_rank, final_rank in (
                ("frozen", initial_bank_rank, final_bank_rank),
                ("unary", initial_unary_rank, final_unary_rank)):
            for top in (1, 3, 5, 10):
                row[f"initial_{family}_top{top}_coverage"] = bool(
                    (initial_rank[agent] <= top).any().cpu())
                row[f"final_{family}_top{top}_coverage"] = bool(
                    (final_rank[agent] <= top).any().cpu())
        rows.append(row)
    return rows


def _finite_summary(values):
    value = np.asarray(values, dtype=np.float64)
    if value.size == 0:
        return {"count": 0}
    if not np.isfinite(value).all():
        raise FloatingPointError("non-finite audit result")
    return {
        "count": int(value.size), "mean": float(value.mean()),
        "std": float(value.std()), "median": float(np.median(value)),
        "p10": float(np.quantile(value, 0.10)),
        "p90": float(np.quantile(value, 0.90)),
    }


def _summarize_group(rows):
    if not rows:
        return {"agent_observations": 0}
    scalar = (
        "initial_unique_count", "initial_duplicate_count",
        "initial_unique_ratio", "final_unique_count",
        "final_duplicate_count", "candidate_changed_fraction",
        "candidate_bank_oracle_error", "initial_slot_oracle_error",
        "final_selected_goal_oracle_error", "trajectory_minFDE",
        "trajectory_best_slot_goal_error", "gt_oracle_frozen_rank",
        "gt_oracle_unary_rank", "mean_initial_frozen_rank",
        "mean_final_frozen_rank", "mean_initial_unary_rank",
        "mean_final_unary_rank", "refinement_frozen_rank_delta",
        "refinement_unary_rank_delta")
    result = {
        "agent_observations": len(rows),
        "scalars": {name: _finite_summary([row[name] for row in rows])
                    for name in scalar},
    }
    for phase in ("initial", "final"):
        for family in ("frozen", "unary"):
            for top in (1, 3, 5, 10):
                name = f"{phase}_{family}_top{top}_coverage"
                result[name] = float(np.mean([row[name] for row in rows]))
    result["error_decomposition"] = {
        "finite_slot_loss": _finite_summary([
            row["initial_slot_oracle_error"] -
            row["candidate_bank_oracle_error"] for row in rows]),
        "refinement_delta": _finite_summary([
            row["final_selected_goal_oracle_error"] -
            row["initial_slot_oracle_error"] for row in rows]),
        "diffusion_gap": _finite_summary([
            row["trajectory_minFDE"] -
            row["final_selected_goal_oracle_error"] for row in rows]),
        "trajectory_vs_bank_gap": _finite_summary([
            row["trajectory_minFDE"] -
            row["candidate_bank_oracle_error"] for row in rows]),
    }
    return result


def _summarize_records(rows):
    result = {"overall": _summarize_group(rows), "strata": {}}
    for field in ("edge_class", "degree_bin", "agent_count_bin"):
        groups = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        result["strata"][field] = {
            label: _summarize_group(group)
            for label, group in sorted(groups.items())}
    return result


def _summarize_metric_runs(runs):
    result = {}
    for stratum in ("overall", "E=0", "E>0"):
        result[stratum] = {}
        for metric in AUDIT_METRICS:
            result[stratum][metric] = _finite_summary([
                run[stratum][metric] for run in runs])
    return result


@torch.no_grad()
def evaluate_policy(evaluator, epoch, policy, reference_states):
    net = evaluator.net
    net.eval()
    records = []
    metric_runs = []
    window_counts = defaultdict(int)
    controller = PairedSamplerIntervention(
        policy, reference_states,
        use_cuda=evaluator.device.type == "cuda")
    for seed in SEEDS:
        print(f"POLICY {policy} seed={seed}", flush=True)
        values = defaultdict(lambda: defaultdict(list))
        with isolated_random_seed(
                seed, use_cuda=evaluator.device.type == "cuda"):
            with controller:
                for window_index, (batch_data, batch_id) in enumerate(
                        evaluator.data_loaders["valid"]):
                    inputs, seq_list = net.prepare_inputs(batch_data, batch_id)
                    metric_mask = compute_metric_mask(seq_list)
                    # The cache fixes this graph before initial allocation.
                    edge_count = int(inputs["jdv2_cache"]["edge_index"].shape[1])
                    controller.begin_window(
                        seed, window_index, edge_count, evaluator.device)
                    with evaluator._autocast_context():
                        prediction, auxiliary = net.forward(
                            inputs, if_test=True)
                    controller.end_window()
                    edge_class = "E>0" if edge_count else "E=0"
                    window_counts[(seed, edge_class)] += 1
                    for metric in AUDIT_METRICS:
                        metric_value = net.compute_model_metrics(
                            metric_name=metric, predictions=prediction,
                            metric_mask=metric_mask,
                            all_aux_outputs=auxiliary, inputs=inputs,
                            obs_length=net.args.obs_length)
                        values["overall"][metric].extend(metric_value)
                        values[edge_class][metric].extend(metric_value)
                    records.extend(_agent_records(
                        net, inputs, prediction, auxiliary, metric_mask,
                        policy, seed, window_index))
                    del inputs, prediction, auxiliary, seq_list
        metric_runs.append({
            "seed": int(seed),
            **{
                stratum: {
                    metric: float(np.mean(items))
                    for metric, items in metric_values.items()}
                for stratum, metric_values in values.items()
            },
        })
    return {
        "metric_runs": metric_runs,
        "metric_summary": _summarize_metric_runs(metric_runs),
        "candidate_summary": _summarize_records(records),
        "window_counts": {
            str(seed): {name: window_counts[(seed, name)]
                        for name in ("E=0", "E>0")}
            for seed in SEEDS},
        "paired_rng_checks": dict(controller.boundary_checks),
    }, records


def _extract_reference_candidate(reference):
    overall = reference["candidate_sampler"]["overall"]
    scalar = overall["scalars"]
    return {
        "initial_unique_count": scalar[
            "initial_unique_candidates"]["mean"],
        "initial_frozen_top1_coverage": overall[
            "initial_top1_coverage"],
        "initial_frozen_top3_coverage": overall[
            "initial_top3_coverage"],
        "initial_frozen_top5_coverage": overall[
            "initial_top5_coverage"],
        "candidate_bank_oracle_error": scalar[
            "candidate_bank_oracle_error"]["mean"],
        "initial_slot_oracle_error": scalar[
            "initial_goal_oracle_error"]["mean"],
        "final_selected_goal_oracle_error": scalar[
            "selected_goal_oracle_error"]["mean"],
        "trajectory_minFDE": scalar["trajectory_minFDE"]["mean"],
    }


def baseline_reproduction_gate(policy_a, metric_reference,
                               candidate_reference, tolerance=1e-10):
    checks = {}
    reference_runs = {
        int(run["seed"]): run for run in metric_reference["runs"]}
    for run in policy_a["metric_runs"]:
        seed = int(run["seed"])
        for metric in AUDIT_METRICS:
            actual = run["overall"][metric]
            expected = reference_runs[seed][metric]
            checks[f"seed_{seed}_{metric}"] = {
                "actual": actual, "expected": expected,
                "absolute_error": abs(actual - expected),
            }
    actual_overall = policy_a["candidate_summary"]["overall"]
    scalar = actual_overall["scalars"]
    actual_candidate = {
        "initial_unique_count": scalar["initial_unique_count"]["mean"],
        "initial_frozen_top1_coverage": actual_overall[
            "initial_frozen_top1_coverage"],
        "initial_frozen_top3_coverage": actual_overall[
            "initial_frozen_top3_coverage"],
        "initial_frozen_top5_coverage": actual_overall[
            "initial_frozen_top5_coverage"],
        "candidate_bank_oracle_error": scalar[
            "candidate_bank_oracle_error"]["mean"],
        "initial_slot_oracle_error": scalar[
            "initial_slot_oracle_error"]["mean"],
        "final_selected_goal_oracle_error": scalar[
            "final_selected_goal_oracle_error"]["mean"],
        "trajectory_minFDE": scalar["trajectory_minFDE"]["mean"],
    }
    for name, actual in actual_candidate.items():
        expected = candidate_reference[name]
        checks[f"candidate_{name}"] = {
            "actual": actual, "expected": expected,
            "absolute_error": abs(actual - expected),
        }
    passed = all(value["absolute_error"] <= tolerance
                 for value in checks.values())
    return {"passed": passed, "tolerance": tolerance, "checks": checks}


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metric-reference", required=True)
    parser.add_argument("--candidate-reference", required=True)
    parser.add_argument("--gdts-reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-agent-output", required=True)
    parser.add_argument("--device", default="cuda:0")
    cli = parser.parse_args()

    output = Path(cli.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    evaluator, epoch = _active_evaluator(
        cli.config, cli.checkpoint, output.parent / "runtime", cli.device)
    if epoch != 13 or not evaluator.net.strict_no_z:
        raise RuntimeError("audit requires strict no-z epoch-13 checkpoint")
    checkpoint = torch.load(cli.checkpoint, map_location="cpu")
    architecture = checkpoint.get("architecture_config", {})
    if architecture.get("architecture_variant") != "strict_no_z":
        raise RuntimeError("checkpoint architecture is not strict_no_z")

    result = {
        "status": "RUNNING_POLICY_A_GATE",
        "protocol": {
            "split": "valid", "windows": 139, "seeds": list(SEEDS),
            "samples": 20, "candidates": 21,
            "sampling_policy": "stochastic deployed policy (not MAP)",
            "rng_pairing": (
                "Policy A captures full Python/NumPy/CPU Torch/all-CUDA "
                "states at pre-initial, pre-refinement, pre-diffusion, and "
                "window-end boundaries; B/C restore the A states at every "
                "downstream boundary. B uses an explicit per-window CUDA "
                "generator for Gumbel-Top-P."),
        },
        "provenance": {
            "checkpoint": str(Path(cli.checkpoint).resolve()),
            "checkpoint_epoch": int(epoch),
            "checkpoint_sha256": file_sha256(cli.checkpoint),
            "architecture_config": architecture,
            "cache_manifest_hash": checkpoint.get("cache_manifest_hash"),
            "source_checkpoint_hash": checkpoint.get(
                "source_checkpoint_hash"),
            "metric_reference": str(Path(cli.metric_reference).resolve()),
            "metric_reference_sha256": file_sha256(cli.metric_reference),
            "candidate_reference": str(
                Path(cli.candidate_reference).resolve()),
            "candidate_reference_sha256": file_sha256(
                cli.candidate_reference),
            "gdts_reference": str(Path(cli.gdts_reference).resolve()),
            "gdts_reference_sha256": file_sha256(cli.gdts_reference),
        },
        "policies": {},
    }
    _write_json(output, result)
    reference_states: Dict[tuple, Dict[str, RNGSnapshot]] = {}
    all_records = []

    policy_a, records = evaluate_policy(
        evaluator, epoch, POLICIES[0], reference_states)
    result["policies"][POLICIES[0]] = policy_a
    all_records.extend(records)
    metric_reference = json.loads(Path(cli.metric_reference).read_text())
    candidate_reference_raw = json.loads(
        Path(cli.candidate_reference).read_text())
    gate = baseline_reproduction_gate(
        policy_a, metric_reference,
        _extract_reference_candidate(candidate_reference_raw))
    result["baseline_reproduction_gate"] = gate
    if not gate["passed"]:
        result["status"] = "STOPPED_BASELINE_REPRODUCTION_FAILED"
        _write_json(output, result)
        raise RuntimeError("Policy A failed the baseline reproduction gate")
    result["status"] = "POLICY_A_GATE_PASSED"
    _write_json(output, result)

    for policy in POLICIES[1:]:
        policy_result, records = evaluate_policy(
            evaluator, epoch, policy, reference_states)
        result["policies"][policy] = policy_result
        all_records.extend(records)
        result["status"] = f"COMPLETED_{policy.upper()}"
        _write_json(output, result)

    expected_windows = len(SEEDS) * 139
    for policy in POLICIES[1:]:
        paired = result["policies"][policy]["paired_rng_checks"]
        for boundary in ("pre_refinement", "pre_diffusion", "window_end"):
            if paired.get(boundary) != expected_windows:
                raise RuntimeError(
                    f"incomplete RNG checks for {policy}/{boundary}: {paired}")

    result["status"] = "AUDIT_COMPLETE"
    _write_json(output, result)
    per_agent = Path(cli.per_agent_output).resolve()
    with per_agent.open("w") as handle:
        for record in all_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"WROTE {output}", flush=True)
    print(f"WROTE {per_agent}", flush=True)


if __name__ == "__main__":
    main()
