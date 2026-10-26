#!/usr/bin/env python3
"""Complete paired inference seeds for the frozen ZARA1 pilot arms."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from ebjd.config import build_model, load_config
from ebjd.proposals import NativeProposalBank
from run_zara1_native_prior_pilot import atomic_json, evaluate, loader, sha256


METRICS = ("minADE", "minFDE", "JADE", "JFDE")


def load_arm(config: dict, checkpoint: Path, expected_sha: str, device):
    if sha256(checkpoint) != expected_sha:
        raise ValueError(f"checkpoint SHA256 mismatch: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(config, proposal_conditioned=True).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model


def aggregate(records: list[dict]) -> dict:
    output = {}
    for arm in ("Control", "Candidate"):
        arm_records = [record[arm] for record in records]
        output[arm] = {"mean": {}, "sample_standard_deviation": {}}
        for key in METRICS:
            values = [record["metrics"][key] for record in arm_records]
            output[arm]["mean"][key] = statistics.fmean(values)
            output[arm]["sample_standard_deviation"][key] = statistics.stdev(values)
        for key in ("recall_at_0_5m", "goal_correction_p50_m",
                    "goal_correction_p95_m"):
            values = [record[key] for record in arm_records]
            output[arm]["mean"][key] = statistics.fmean(values)
            output[arm]["sample_standard_deviation"][key] = statistics.stdev(values)
        output[arm]["N_eq_1_mean"] = {
            key: statistics.fmean(record["N_eq_1"][key] for record in arm_records)
            for key in METRICS}
        output[arm]["N_ge_2_mean"] = {
            key: statistics.fmean(record["N_ge_2"][key] for record in arm_records)
            for key in METRICS}
    output["paired_candidate_minus_control"] = {
        key: [
            record["Candidate"]["metrics"][key]
            - record["Control"]["metrics"][key]
            for record in records]
        for key in METRICS}
    output["candidate_improvement_seed_count"] = {
        key: sum(value < 0.0 for value in values)
        for key, values in output["paired_candidate_minus_control"].items()}
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--seed2035-result", required=True)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--control-sha256", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--proposal-bank-dir", required=True)
    parser.add_argument("--gdts-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2036, 2037, 2038, 2039])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_config(args.model_config)
    device = torch.device(args.device)
    output = Path(args.output).resolve()
    per_seed = output / "per_seed"
    per_seed.mkdir(parents=True, exist_ok=True)
    control = load_arm(config, Path(args.control_checkpoint), args.control_sha256, device)
    candidate = load_arm(config, Path(args.candidate_checkpoint), args.candidate_sha256, device)
    validation_loader = loader(
        config, "validation", torch.Generator().manual_seed(99173))

    seed2035 = json.loads(Path(args.seed2035_result).read_text(encoding="utf-8"))
    records = [{
        "seed": 2035,
        "Control": seed2035["arms"]["Control"]["final_seed2035"],
        "Candidate": seed2035["arms"]["Candidate"]["final_seed2035"],
        "proposal_bank_sha256": seed2035["identity"]["validation_proposal_bank_sha256"],
    }]
    for seed in args.seeds:
        target = per_seed / f"seed_{seed}.json"
        if target.exists():
            record = json.loads(target.read_text(encoding="utf-8"))
            if record.get("status") != "COMPLETE" or record.get("seed") != seed:
                raise ValueError(f"invalid existing seed receipt: {target}")
            records.append(record)
            continue
        bank_path = Path(args.proposal_bank_dir) / f"gdts_proposal_B_seed{seed}.npz"
        bank = NativeProposalBank(
            bank_path, expected_checkpoint_sha256=args.gdts_sha256,
            expected_manifest_sha256=args.manifest_sha256,
            expected_split="validation")
        record = {
            "status": "COMPLETE", "seed": seed,
            "proposal_bank": str(bank_path.resolve()),
            "proposal_bank_sha256": bank.sha256,
            "Control": evaluate(control, validation_loader, bank, device, 0.0, seed, 20, 20),
            "Candidate": evaluate(candidate, validation_loader, bank, device, 1.0, seed, 20, 20),
        }
        atomic_json(target, record)
        records.append(record)
        print(json.dumps({
            "seed": seed,
            "control": record["Control"]["metrics"],
            "candidate": record["Candidate"]["metrics"],
        }, sort_keys=True), flush=True)
    records.sort(key=lambda record: record["seed"])
    if [record["seed"] for record in records] != [2035, 2036, 2037, 2038, 2039]:
        raise ValueError("five-seed result is incomplete")
    result = {
        "schema": "ebjd-native-prior-conditional-bridge-five-seed-v1",
        "status": "COMPLETE", "seeds": [record["seed"] for record in records],
        "worlds": 20, "ddim_steps": 20, "records": records,
        "summary": aggregate(records),
    }
    atomic_json(output / "FIVE_SEED_RESULT.json", result)
    print(json.dumps(result["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
