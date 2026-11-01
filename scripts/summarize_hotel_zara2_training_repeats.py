#!/usr/bin/env python3
"""Summarize HOTEL/ZARA2 repeats with correctly nested random seeds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
import subprocess
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
ASSETS = ROOT / "RSJG_EBJD_assets"
METRICS = ("minADE", "minFDE", "JADE", "JFDE")
FOLDS = ("hotel", "zara2")
TRAINING_SEEDS = (3101, 3102, 3103)
SOURCE_RUNS = {
    "hotel": ASSETS / "experiments/hotel_unified_coupling_bridge_v1_20261030_211601",
    "zara2": ASSETS / "experiments/zara2_unified_coupling_bridge_v1_20261029_094715",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def metric_rows(fold: str, training_seed: int, root: Path) -> tuple[list[dict], Path]:
    if training_seed == 3101:
        path = SOURCE_RUNS[fold] / "reports/PER_SEED_METRICS.csv"
        original = read_csv(path)
        seed_key = "seed"
    else:
        path = root / fold / str(training_seed) / "reports/PER_SEED_METRICS.csv"
        original = read_csv(path)
        seed_key = "inference_seed"
    normalized = []
    for row in original:
        output = {"training_seed": training_seed, "inference_seed": int(row[seed_key])}
        for method in ("G", "H", "GDTS"):
            for metric in METRICS:
                output[f"{method}_{metric}"] = float(row[f"{method}_{metric}"])
        normalized.append(output)
    return normalized, path


def checkpoint_identity(fold: str, seed: int, root: Path) -> dict:
    if seed != 3101:
        path = root / fold / str(seed) / "reports/CHECKPOINT_MANIFEST.json"
        return json.loads(path.read_text()) | {"source": str(path)}
    run = SOURCE_RUNS[fold]
    path = run / "training/FINAL_EVALUATION.json"
    final = json.loads(path.read_text())
    selected_epoch = next(
        int(row["epoch"]) for row in read_csv(run / "training/checkpoint_index.csv")
        if row["sha256"] == final["selected_checkpoint_sha256"])
    return {
        "selected_epoch": selected_epoch,
        "selection_identity": final["selection_identity"],
        "selected_checkpoint": final["selected_checkpoint"],
        "selected_checkpoint_sha256": final["selected_checkpoint_sha256"],
        "source": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    args = parser.parse_args()
    root = Path(args.experiment_root).resolve()
    summary, figures = root / "summary", root / "summary/figures"
    figures.mkdir(parents=True, exist_ok=True)

    inference_rows: list[dict] = []
    identities: dict[tuple[str, int], dict] = {}
    sources: dict = {"experiment_root": str(root), "folds": {}}
    for fold in FOLDS:
        sources["folds"][fold] = {}
        reference = None
        for training_seed in TRAINING_SEEDS:
            rows, path = metric_rows(fold, training_seed, root)
            if [r["inference_seed"] for r in rows] != list(range(2035, 2040)):
                raise RuntimeError(f"{fold}/{training_seed}: inference seeds are incomplete")
            for row in rows:
                inference_rows.append({"fold": fold.upper(), **row})
            identities[(fold, training_seed)] = checkpoint_identity(fold, training_seed, root)
            sources["folds"][fold][str(training_seed)] = {
                "metrics": str(path), "metrics_sha256": sha256(path),
                **identities[(fold, training_seed)],
            }
            current = tuple(tuple(r[f"GDTS_{m}"] for m in METRICS) for r in rows)
            if reference is None:
                reference = current
            elif current != reference:
                raise RuntimeError(f"{fold}: matched GDTS changed across repeats")
    write_csv(summary / "INFERENCE_SEED_METRICS.csv", inference_rows)
    write_json(summary / "SOURCE_MANIFEST.json", sources)

    training_rows: list[dict] = []
    contracts: dict[str, dict] = {f: {} for f in FOLDS}
    means: dict[tuple[str, int, str, str], float] = {}
    for fold in FOLDS:
        for training_seed in TRAINING_SEEDS:
            rows = [r for r in inference_rows if r["fold"] == fold.upper()
                    and r["training_seed"] == training_seed]
            for method in ("GDTS", "G", "H"):
                for metric in METRICS:
                    values = [float(r[f"{method}_{metric}"]) for r in rows]
                    means[(fold, training_seed, method, metric)] = statistics.mean(values)
                    training_rows.append({
                        "fold": fold.upper(), "training_seed": training_seed,
                        "method": method, "metric": metric,
                        "inference_mean": statistics.mean(values),
                        "inference_sd_ddof1": statistics.stdev(values),
                        "training_seed_mean": "", "training_seed_sd_ddof1": "",
                        "selected_epoch": identities[(fold, training_seed)]["selected_epoch"],
                        "selection_identity": identities[(fold, training_seed)]["selection_identity"],
                    })
            metric_pass = {m: means[(fold, training_seed, "H", m)] <=
                                means[(fold, training_seed, "GDTS", m)] for m in METRICS}
            joint_gain = {m: means[(fold, training_seed, "H", m)] <
                              means[(fold, training_seed, "G", m)] for m in ("JADE", "JFDE")}
            contracts[fold][str(training_seed)] = {
                "metric_pass_H_vs_GDTS": metric_pass,
                "four_metric_contract_pass": all(metric_pass.values()),
                "joint_gain_H_vs_G": joint_gain,
                "joint_gain_pass": all(joint_gain.values()),
                "minFDE_exactly_preserved_all_inference_seeds": all(
                    r["H_minFDE"] == r["G_minFDE"] for r in rows),
                "H_minus_GDTS": {m: means[(fold, training_seed, "H", m)] -
                                      means[(fold, training_seed, "GDTS", m)] for m in METRICS},
                "H_minus_G": {m: means[(fold, training_seed, "H", m)] -
                                  means[(fold, training_seed, "G", m)] for m in METRICS},
                "selected_epoch": identities[(fold, training_seed)]["selected_epoch"],
                "selection_identity": identities[(fold, training_seed)]["selection_identity"],
            }
    # Correct nesting: each value below is already a five-inference-seed mean.
    for fold in FOLDS:
        for method in ("G", "H"):
            for metric in METRICS:
                values = [means[(fold, s, method, metric)] for s in TRAINING_SEEDS]
                training_rows.append({
                    "fold": fold.upper(), "training_seed": "ALL",
                    "method": method, "metric": metric,
                    "inference_mean": "", "inference_sd_ddof1": "",
                    "training_seed_mean": statistics.mean(values),
                    "training_seed_sd_ddof1": statistics.stdev(values),
                    "selected_epoch": "", "selection_identity": "",
                })
    write_csv(summary / "TRAINING_SEED_METRICS.csv", training_rows)
    write_json(summary / "CONTRACT_BY_TRAINING_SEED.json", contracts)

    budget = json.loads((root / "GPU_BUDGET.json").read_text())
    costs = [{"stage": x["stage"], "gpu_wall_seconds": x["seconds"], "status": x["status"]}
             for x in budget["stages"]]
    costs.append({"stage": "TOTAL_NEW_REPEATS", "gpu_wall_seconds": budget["consumed_seconds"],
                  "status": "COMPLETE"})
    write_csv(summary / "COMPUTE_COSTS.csv", costs)

    edge, joint, distribution = [], [], []
    for seed in TRAINING_SEEDS:
        for metric in ("minADE", "minFDE"):
            edge.append({"training_seed": seed, "metric": metric,
                         "H_minus_GDTS": means[("hotel", seed, "H", metric)] -
                         means[("hotel", seed, "GDTS", metric)]})
        for fold in FOLDS:
            for metric in ("JADE", "JFDE"):
                joint.append({"fold": fold.upper(), "training_seed": seed, "metric": metric,
                              "H_minus_G": means[(fold, seed, "H", metric)] -
                              means[(fold, seed, "G", metric)]})
            for metric in METRICS:
                distribution.append({"fold": fold.upper(), "training_seed": seed,
                                     "metric": metric, "H": means[(fold, seed, "H", metric)],
                                     "GDTS": means[(fold, seed, "GDTS", metric)]})
    write_csv(figures / "hotel_edge_deltas.csv", edge)
    write_csv(figures / "pairing_joint_deltas.csv", joint)
    write_csv(figures / "training_seed_distributions.csv", distribution)
    shutil.copy2(PAPER / "scripts/plot_hotel_zara2_training_repeats.py", figures / "plot_results.py")
    subprocess.run([str(ROOT / ".conda/rsjg/bin/python"), str(figures / "plot_results.py"),
                    "--summary-dir", str(summary)], check=True)

    hotel = contracts["hotel"]
    zara2 = contracts["zara2"]
    hotel_minade = [hotel[str(s)]["H_minus_GDTS"]["minADE"] for s in TRAINING_SEEDS]
    selected = {f: [contracts[f][str(s)]["selected_epoch"] for s in TRAINING_SEEDS] for f in FOLDS}
    report = [
        "# HOTEL/ZARA2 独立训练 seed 复验", "", "## 结论", "",
        f"- HOTEL：H 相对 G 的 JADE/JFDE 同时改善 {sum(hotel[str(s)]['joint_gain_pass'] for s in TRAINING_SEEDS)}/3 个训练 seed；四指标合同通过 {sum(hotel[str(s)]['four_metric_contract_pass'] for s in TRAINING_SEEDS)}/3。",
        f"- ZARA2：H 相对 G 的 JADE/JFDE 同时改善 {sum(zara2[str(s)]['joint_gain_pass'] for s in TRAINING_SEEDS)}/3 个训练 seed；四指标合同通过 {sum(zara2[str(s)]['four_metric_contract_pass'] for s in TRAINING_SEEDS)}/3。",
        f"- HOTEL H−GDTS minADE（3101/3102/3103）：{', '.join(f'{x:+.6f}' for x in hotel_minade)} m。",
        f"- 选中 epoch：HOTEL {selected['hotel']}；ZARA2 {selected['zara2']}。",
        f"- 本轮新增 GPU 墙钟：{budget['consumed_seconds']/3600:.3f} h / 48 h。", "",
        "先在每个 training seed 内平均五个 inference seeds，再对三个 training-seed 均值计算均值与样本标准差（ddof=1）。共同 matched GDTS 未被重复当作三个独立基线训练。", "",
        "## 证据边界", "",
        "结论仅覆盖固定终点先验、地图初始化和 proposal bank 下的主模型训练随机性；三个 training seeds 不是十五个独立训练。其余三个 fold 仍只有一个 training seed。", "",
        "## 唯一优先建议", "",
        "若两折的配对联合收益在三个 training seed 中均保持，优先补齐其余三折训练重复；否则先进入关键配对消融。", "",
    ]
    (summary / "REPORT_ZH.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
