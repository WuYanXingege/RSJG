#!/usr/bin/env python3
"""Export frozen Candidate references and evaluate MPJC endpoint coupling."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ebjd.config import build_model, load_config
from ebjd.coupling import couple_bank, relative_pairing_change
from ebjd.data import SceneManifestDataset, collate_scenes
from ebjd.proposals import NativeProposalBank, file_sha256
from ebjd.sampling import predict
from run_zara1_native_prior_pilot import atomic_json, loader


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def endpoint_summary(
    endpoints: np.ndarray, truth: np.ndarray, valid: np.ndarray,
    selector: np.ndarray,
) -> dict:
    selected_valid = valid & selector[:, None]
    distances = np.linalg.norm(
        endpoints.astype(np.float64) - truth[:, None].astype(np.float64), axis=-1)
    best = np.where(selected_valid[:, None], distances, np.inf).min(1)[selected_valid]
    joint = []
    for scene in np.flatnonzero(selector):
        mask = valid[scene]
        joint.append(distances[scene][:, mask].mean(1).min())
    return {
        "minFDE": float(best.mean()), "JFDE": float(np.mean(joint)),
        "Recall@0.5m": float(np.mean(best <= 0.5)),
        "agent_count": int(selected_valid.sum()), "scene_count": int(selector.sum()),
    }


def summaries(endpoints, truth, valid) -> dict:
    counts = valid.sum(1)
    groups = {
        "all": np.ones(len(valid), dtype=bool),
        "N=1": counts == 1,
        "N>=2": counts >= 2,
    }
    return {
        name: endpoint_summary(endpoints, truth, valid, selector)
        for name, selector in groups.items()
    }


def load_reference(config: dict, path: Path, expected_sha: str, device):
    actual = file_sha256(path)
    if actual != expected_sha:
        raise ValueError(f"Candidate checkpoint SHA mismatch: {actual}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(config, proposal_conditioned=True).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model


def export_seed(args, config, reference, device, seed: int) -> dict:
    bank_path = Path(args.bank_pattern.format(seed=seed)).resolve()
    bank = NativeProposalBank(
        bank_path, expected_checkpoint_sha256=args.gdts_sha256,
        expected_manifest_sha256=args.manifest_sha256,
        expected_split=args.split)
    seed_dir = Path(args.output).resolve() / f"seed_{seed}"
    receipt_path = seed_dir / "RESULT.json"
    coupled_path = seed_dir / "MPJC_ENDPOINT_BANK.npz"
    if receipt_path.exists() and coupled_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt.get("status") == "COMPLETE"
                and receipt.get("seed") == seed
                and receipt.get("bank_sha256") == file_sha256(coupled_path)):
            return receipt
        raise ValueError(f"invalid existing MPJC receipt: {receipt_path}")

    proposals = bank.endpoints.numpy()
    references = np.zeros_like(proposals)
    if args.split == "validation":
        evaluation_loader = loader(
            config, "validation", torch.Generator().manual_seed(99173))
    else:
        # Reference export is identity ordered and never augmented/shuffled.
        # The subsequent bridge training owns its independent loader RNG.
        evaluation_loader = DataLoader(
            SceneManifestDataset(config["paths"]["train_manifest"]),
            batch_size=1, shuffle=False, collate_fn=collate_scenes,
            num_workers=0)
    generator = torch.Generator(device=device).manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    cursor = 0
    with torch.no_grad():
        for cpu_batch in evaluation_loader:
            if len(cpu_batch.scene_ids) != 1:
                raise ValueError("reference export requires one scene per batch")
            source = bank._index[cpu_batch.scene_ids[0]]
            if source != cursor:
                raise ValueError("manifest and bank scene ordering differ")
            native = bank.batch(cpu_batch).to(device)
            batch = cpu_batch.to(device)
            _, goals = predict(
                reference, batch.observed, batch.semantic_maps, batch.valid,
                worlds=20, steps=20, generator=generator,
                proposal_goals=native, proposal_eta=1.0)
            count = int(cpu_batch.valid[0].sum())
            references[cursor, :, :count] = goals[0, :, :count].float().cpu().numpy()
            cursor += 1
    if cursor != len(bank.scene_ids):
        raise RuntimeError("reference export did not cover the complete manifest")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gpu_seconds = time.perf_counter() - started

    coupled, permutation = couple_bank(proposals, references, bank.valid.numpy())
    valid = bank.valid.numpy()
    truth = bank.truth.numpy()
    # Exact indexed gather proves preservation even when coordinates repeat.
    gathered = np.take_along_axis(
        proposals, permutation[..., None], axis=1)
    # Padded endpoint slots retain the source bank's NaNs.  Equal-position
    # NaNs are identical padding semantics, not a changed candidate value.
    if not np.array_equal(coupled, gathered, equal_nan=True):
        raise RuntimeError("coupled endpoints are not an exact proposal gather")
    for scene in range(len(valid)):
        for agent in np.flatnonzero(valid[scene]):
            if not np.array_equal(
                np.sort(permutation[scene, :, agent]), np.arange(20)):
                raise RuntimeError("an agent assignment is not a full permutation")
    pairing = [
        relative_pairing_change(permutation[scene], valid[scene])
        for scene in range(len(valid)) if int(valid[scene].sum()) >= 2]
    metadata = {
        "schema": "ebjd-mpjc-endpoint-bank-v1", "seed": seed,
        "worlds": 20, "ddim_steps": 20, "split": args.split,
        "coordinate_unit": "m", "candidate_checkpoint": str(Path(args.candidate_checkpoint).resolve()),
        "candidate_checkpoint_sha256": args.candidate_sha256,
        "native_proposal_bank": str(bank_path), "native_proposal_bank_sha256": bank.sha256,
        "manifest_sha256": args.manifest_sha256,
        "reference_uses_ground_truth": False,
        "coupling_uses_ground_truth": False,
        "permutation_semantics": "output_world_to_native_proposal_index",
    }
    atomic_npz(
        coupled_path, native_endpoints=proposals, reference_endpoints=references,
        coupled_endpoints=coupled, permutation=permutation, truth=truth,
        valid=valid, scene_ids=np.asarray(bank.scene_ids),
        agent_ids=np.asarray(bank.agent_ids),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    path_summaries = {
        "G_native": summaries(proposals, truth, valid),
        "R_reference": summaries(references, truth, valid),
        "H_coupled": summaries(coupled, truth, valid),
    }
    receipt = {
        "schema": "ebjd-mpjc-frozen-endpoint-result-v1", "status": "COMPLETE",
        "seed": seed, "worlds": 20, "ddim_steps": 20,
        "candidate_checkpoint_sha256": args.candidate_sha256,
        "native_bank_path": str(bank_path), "native_bank_sha256": bank.sha256,
        "bank_path": str(coupled_path), "bank_sha256": file_sha256(coupled_path),
        "summaries": path_summaries,
        "H_minus_G": {
            group: {
                metric: path_summaries["H_coupled"][group][metric]
                        - path_summaries["G_native"][group][metric]
                for metric in ("minFDE", "JFDE", "Recall@0.5m")
            } for group in ("all", "N=1", "N>=2")
        },
        "coverage_exact": True,
        "pairing": {
            "multi_scene_count": len(pairing),
            "changed_scene_count": sum(item["changed_pair_count"] > 0 for item in pairing),
            "agent_pair_count": sum(item["pair_count"] for item in pairing),
            "changed_agent_pair_count": sum(item["changed_pair_count"] for item in pairing),
            "mean_relative_change": float(np.mean([
                item["mean_relative_change"] for item in pairing])),
        },
        "gpu_seconds": gpu_seconds,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--bank-pattern", required=True)
    parser.add_argument("--gdts-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config(args.model_config)
    device = torch.device(args.device)
    reference = load_reference(
        config, Path(args.candidate_checkpoint), args.candidate_sha256, device)
    records = []
    for seed in args.seeds:
        record = export_seed(args, config, reference, device, seed)
        records.append(record)
        print(json.dumps({
            "seed": seed, "gpu_seconds": record["gpu_seconds"],
            "G_N_ge_2": record["summaries"]["G_native"]["N>=2"],
            "H_N_ge_2": record["summaries"]["H_coupled"]["N>=2"],
            "delta": record["H_minus_G"]["N>=2"],
        }, sort_keys=True), flush=True)
    atomic_json(Path(args.output).resolve() / "ENDPOINT_RESULTS.json", {
        "schema": "ebjd-mpjc-frozen-endpoint-results-v1",
        "status": "COMPLETE", "records": records,
        "gpu_seconds": sum(record["gpu_seconds"] for record in records),
    })


if __name__ == "__main__":
    main()
