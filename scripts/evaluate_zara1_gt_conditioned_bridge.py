#!/usr/bin/env python3
"""Atomic five-seed G/H evaluation for the frozen GT-conditioned bridge."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from ebjd.config import build_model, load_config
from ebjd.coupling import MPJCEndpointBank
from ebjd.proposals import file_sha256
from run_zara1_gt_conditioned_bridge import evaluate_mode
from run_zara1_native_prior_pilot import atomic_json


METRICS = ("minADE", "minFDE", "JADE", "JFDE")


def aggregate(records):
    output = {}
    for mode in ("G", "H"):
        rows = [record[mode] for record in records]
        output[mode] = {
            "mean": {key: statistics.fmean(row["metrics"][key] for row in rows) for key in METRICS},
            "sample_standard_deviation": {
                key: statistics.stdev(row["metrics"][key] for row in rows) for key in METRICS},
            "prefix_mean": {
                key: statistics.fmean(row["prefix"][key] for row in rows)
                for key in ("prefix_minADE_11", "prefix_JADE_11")},
            "prefix_sample_standard_deviation": {
                key: statistics.stdev(row["prefix"][key] for row in rows)
                for key in ("prefix_minADE_11", "prefix_JADE_11")},
            "recall_at_0_5m_mean": statistics.fmean(row["recall_at_0_5m"] for row in rows),
        }
    output["H_minus_G_per_seed"] = {
        key: [record["H"]["metrics"][key] - record["G"]["metrics"][key] for record in records]
        for key in METRICS}
    output["H_improvement_seed_count"] = {
        key: sum(value < 0 for value in values)
        for key, values in output["H_minus_G_per_seed"].items()}
    return output


def bank_path(args, seed):
    return Path(args.bank_2035 if seed == 2035 else args.bank_pattern.format(seed=seed))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--bank-2035", required=True)
    parser.add_argument("--bank-pattern", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2035,2036,2037,2038,2039])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    if file_sha256(checkpoint) != args.checkpoint_sha256:
        raise ValueError("GT-conditioned checkpoint SHA256 mismatch")
    config = load_config(args.model_config)
    device = torch.device(args.device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(config, proposal_conditioned=True).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    output = Path(args.output).resolve()
    per_seed = output / "per_seed"
    per_seed.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in args.seeds:
        target = per_seed / f"seed_{seed}.json"
        if target.exists():
            record = json.loads(target.read_text(encoding="utf-8"))
            if record.get("status") != "COMPLETE" or record.get("seed") != seed:
                raise ValueError(f"invalid existing seed receipt: {target}")
            records.append(record)
            continue
        bank = MPJCEndpointBank(
            str(bank_path(args, seed)),
            expected_candidate_sha256=args.candidate_sha256,
            expected_manifest_sha256=args.manifest_sha256,
            expected_split="validation")
        record = {
            "status": "COMPLETE", "seed": seed,
            "checkpoint_sha256": args.checkpoint_sha256,
            "mpjc_bank": bank.path, "mpjc_bank_sha256": bank.sha256,
            "G": evaluate_mode(model, config, bank, "native", device, seed),
            "H": evaluate_mode(model, config, bank, "coupled", device, seed),
        }
        atomic_json(target, record)
        records.append(record)
        print(json.dumps({
            "seed": seed, "G": record["G"]["metrics"],
            "H": record["H"]["metrics"],
        }, sort_keys=True), flush=True)
    records.sort(key=lambda row: row["seed"])
    expected = sorted(args.seeds)
    if [row["seed"] for row in records] != expected:
        raise RuntimeError("requested seed set is incomplete")
    result = {
        "schema": "ebjd-zara1-gt-conditioned-bridge-five-seed-v1",
        "status": "COMPLETE", "seeds": expected,
        "checkpoint_sha256": args.checkpoint_sha256,
        "records": records, "summary": aggregate(records),
        "gpu_seconds": sum(
            row[mode]["elapsed_seconds"] for row in records for mode in ("G", "H")),
    }
    atomic_json(output / "FIVE_SEED_RESULT.json", result)
    print(json.dumps(result["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
