"""Seal development K20 permutations, score the unchanged trajectories, and report."""

from __future__ import annotations

import hashlib
import math
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from bcep.contracts import require
from bcep_formal.data import FormalData
from direct_k20.common import ATOL, RTOL, SOURCE, digest, immutable, read, window_key, write
from direct_k20.interaction_diagnosis.assets import ARCHIVE, save_npz
from direct_k20.interaction_diagnosis.statistics import (
    greedy_observed_pairs, group_for_agents, partition_jfde,
)
from direct_k20.interaction_eval import _row_metrics
from ebjd.metrics import hard_metric_statistics
from joint_flow_fourfold.report import csvwrite, public_metadata, read_csv

from .matcher import Matcher, hard_permutation, permute_paths, window_features


DEV = "selection_dev"
SEED = 2035
FOLDS = ("hotel", "univ")
K = 20
WINDOWS = {"hotel": 688, "univ": 622}
TARGETS = {"hotel": ("N2_8", 440), "univ": ("N9plus", 61)}
METRICS = ("JADE", "JFDE", "minADE", "minFDE")
TOL = 1e-6
GROUPS = ("all", "N1", "N2_8", "N9plus", "eligible", "changed", "top10",
          "single_group", "all_singleton_multigroup", "rename", "relative")
GROUP_LABEL = dict(zip(
    GROUPS,
    ("全体", "单人", "2–8 人", "N≥9", "可相对换槽", "整场集合真正改变", "S0 高误差前 10%",
     "单组", "多组且各组均为单人", "整场换名", "相对换槽"),
))
DIAG_DOC = SOURCE / "docs/research/continuous_joint_flow/endpoint_diagnosis_20261127"


def _sha_array(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def _bitwise_equal(a, b):
    return np.array_equal(np.ascontiguousarray(a).view(np.uint32),
                          np.ascontiguousarray(b).view(np.uint32))


def _source(ref, index, read_path):
    paths = read_path(index) if read_path is not None else ref.pair(index, SEED)[0]["S0"]
    n = ref.rows[index, SEED]["n"]
    require(paths.dtype == np.float32 and paths.shape == (K, n, 12, 2)
            and np.isfinite(paths).all(), "invalid native S0 complete trajectory")
    return paths


def _check_reference(ref, fold):
    require(ref.purpose == DEV and ref.fold == fold and fold in FOLDS, "development readers only")
    require(list(ref.seeds) == [SEED] and len(ref.indices) == WINDOWS[fold],
            "unexpected original development population")


def _group_labels(n, pairs, group_index):
    """Compare label-independent group membership to the original distance partition."""
    require(group_index.dtype == np.int64 and group_index.shape == (n,), "invalid group index")
    count = n - len(pairs)
    require(np.array_equal(np.unique(group_index), np.arange(count)), "invalid group population")
    freqs = np.bincount(group_index)
    require(np.all(freqs <= 2) and (freqs == 2).sum() == len(pairs)
            and np.all(group_index[pairs[:, 0]] == group_index[pairs[:, 1]]),
            "matcher groups differ from original observed pairing")
    return count


def _inspect_pair(source, candidate, permutation, group_index, pairs):
    n = source.shape[1]
    require(candidate.dtype == source.dtype and candidate.shape == source.shape, "bad output layout")
    require(pairs.dtype == np.int64 and pairs.ndim == 2 and pairs.shape[1] == 2
            and np.all((pairs >= 0) & (pairs < n)) and np.unique(pairs).size == pairs.size,
            "invalid partition pair indexes")
    group_count = _group_labels(n, pairs, group_index)
    require(permutation.dtype == np.int64 and permutation.shape == (group_count, K)
            and np.all(np.sort(permutation, axis=-1) == np.arange(K)), "non-bijective group assignment")
    identity = bool(np.all(permutation == np.arange(K)))
    relative = bool(np.any(permutation != permutation[:1]))
    require(group_count > 1 or identity, "single group must copy its S0 slots literally")
    draw = permute_paths(source, permutation, group_index)
    recovered = permute_paths(candidate, permutation, group_index, inverse=True)
    require(_bitwise_equal(draw, candidate) and _bitwise_equal(recovered, source),
            "complete trajectories do not match the sealed assignment and its inverse bitwise")
    is_changed = relative and sorted(slot.tobytes() for slot in draw) != sorted(
        slot.tobytes() for slot in source)
    return dict(groups_count=group_count, paired_groups=len(pairs), identity=identity,
                relative=relative, global_relabel=not identity and not relative,
                true_change=bool(is_changed))


def _row_identity(ref, index, ordinal, source):
    old_row = ref.rows[index, SEED]
    n = old_row["n"]
    require(window_key(ref.split, index) == old_row["member_sha256"], "old window alignment differs")
    return dict(window=f"window_{ordinal:04d}", file=f"{ordinal:04d}.npz",
                n=n, member_sha256=old_row["member_sha256"], S0_array_sha256=_sha_array(source))


def _load_shard(root, expected, source):
    path = root / expected["file"]
    row = read(path.with_suffix(".json"))
    require(all(row[key] == val for key, val in expected.items())
            and row["sha256"] == digest(path), "frozen development prediction or identity changed")
    with np.load(path, allow_pickle=False) as store:
        require(set(store.files) == {"paths", "permutation", "group_index", "pairs"},
                "sealed tensors differ")
        fields = {name: store[name] for name in store.files}
    checks = _inspect_pair(source, fields["paths"], fields["permutation"],
                           fields["group_index"], fields["pairs"])
    require(all(row[key] == val for key, val in checks.items()), "sealed permutation accounting differs")
    return row, fields


@torch.no_grad()
def export_fold(*, fold: str, device, matcher: Matcher, queries, dev_reference,
                read_endpoints: Callable, output: Path, identity: dict,
                read_path: Callable | None = None):
    """Use only observed frames and S0 predictions, and leave a seal for independent scoring."""
    ref = dev_reference
    _check_reference(ref, fold)
    root = Path(output) / "development" / fold
    root.mkdir(parents=True, exist_ok=True)
    protocol = dict(fold=fold, purpose=DEV, seed=SEED, K=K, frames=12,
                    source_hashes=ref.source_hashes, matcher_identity=identity)
    immutable(root / "EXPORT_CONFIG.json", protocol)
    protocol_sha = digest(root / "EXPORT_CONFIG.json")
    refiner = matcher.eval()
    queries = torch.as_tensor(queries, dtype=torch.float32, device=device).detach()
    reader = FormalData(ref.split, phase="export", verify=False)
    rows = []
    for ordinal, index in enumerate(ref.indices, 1):
        source = _source(ref, index, read_path)
        expected = _row_identity(ref, index, ordinal, source) | {"export_sha256": protocol_sha}
        path = root / expected["file"]
        side = path.with_suffix(".json")
        if side.exists():
            row, _ = _load_shard(root, expected, source)
        else:
            tic = time.monotonic()
            record = ref.split["records"][index]
            require(digest(record["path"]) == record["sha256"], "observed development record changed")
            batch = reader.batch([index], DEV, truth=False)
            n = expected["n"]
            require(batch.valid.shape == (1, n) and bool(batch.valid.all())
                    and not bool(batch.future.count_nonzero()), "export expected observed-only packed data")
            observed = batch.observed[0].numpy()
            goals = np.asarray(read_endpoints(index)["g0"])
            require(goals.shape == (K, n, 2) and goals.dtype == np.float32
                    and np.isfinite(goals).all() and np.allclose(goals, source[..., -1, :],
                                                                 atol=ATOL, rtol=RTOL),
                    "frozen native endpoint does not match S0 slots and complete trajectory")
            pairs = greedy_observed_pairs(observed.astype(np.float64))
            feat = window_features(observed, goals)
            group_index = feat["group_index"]
            group_count = _group_labels(n, pairs, group_index)
            if group_count == 1:
                permutation = np.arange(K, dtype=np.int64)[None]
                candidate = source.copy()
            else:
                score = refiner(torch.from_numpy(feat["row_features"]).to(device),
                                torch.from_numpy(feat["column_features"]).to(device), queries)
                permutation = hard_permutation(score).cpu().numpy()
                candidate = permute_paths(source, permutation, group_index)
            checks = _inspect_pair(source, candidate, permutation, group_index, pairs)
            save_npz(path, paths=candidate, permutation=permutation, group_index=group_index, pairs=pairs)
            row = expected | checks | dict(
                sha256=digest(path), endpoint_max_abs_m=float(np.max(np.abs(goals.astype(np.float64)
                                                                          - source[..., -1, :]))),
                inference_seconds=time.monotonic() - tic,
            )
            immutable(side, row)
        rows.append(row)
    core = dict(fold=fold, purpose=DEV, seed=SEED, windows=len(rows), rows=rows,
                export_sha256=protocol_sha, future_reads=0, K=K)
    path = root / "PREDICTIONS_FROZEN.json"
    if path.exists():
        seal = read(path)
        require(all(seal[key] == val for key, val in core.items()), "completed development export differs")
    else:
        immutable(path, core | {"sealed_unix": time.time()})
    return dict(fold=fold, windows=len(rows), predictions_sha256=digest(path),
                hash_checked_pairs=len(rows), truth_reads=0,
                endpoint_max_abs_m=max(row["endpoint_max_abs_m"] for row in rows),
                inference_seconds=math.fsum(row["inference_seconds"] for row in rows))


def _score_pair(source, candidate, future, valid, pairs):
    native = [hard_metric_statistics(torch.from_numpy(paths)[None], future, valid)
              for paths in (source, candidate)]
    scores = [_row_metrics(stats) for stats in native]
    diff_ade = float((native[1].agent_minade - native[0].agent_minade).abs().max())
    diff_fde = float((native[1].agent_minfde - native[0].agent_minfde).abs().max())
    require(diff_ade == 0 and diff_fde == 0, "native per-person marginal scores changed")
    truth = future[0, :, -1].numpy().astype(np.float64)
    errors = [np.linalg.norm(paths[..., -1, :].astype(np.float64) - truth[None], axis=-1)
              for paths in (source, candidate)]
    jp0, jp1 = (partition_jfde(error, pairs) for error in errors)
    l0, l1 = (float(error.min(axis=0).mean()) for error in errors)
    native_l = abs(scores[1]["L"] - scores[0]["L"])
    require(jp0 == jp1 and l0 == l1 and native_l == 0, "L or same-partition JP is not invariant")
    j0, j1 = (float(error.mean(axis=1).min()) for error in errors)
    require(l0 <= jp0 + 1e-12 and jp0 <= min(j0, j1) + 1e-12, "partition lower bound changed")
    row = {f"{branch}_{metric}_m": value
           for branch, values in zip(("S0", "new"), scores) for metric, value in values.items()}
    row.update({f"delta_{key}_m": scores[1][key]-scores[0][key] for key in (*METRICS, "L", "G")})
    closure = abs(row["delta_JFDE_m"] - row["delta_G_m"])
    require(closure <= 1e-14, "native JFDE delta differs from coordination gap delta")
    row.update(JP_m=jp0, S0_J_minus_JP_m=j0-jp0, new_J_minus_JP_m=j1-jp1,
               native_minADE_max_abs_m=diff_ade, native_minFDE_max_abs_m=diff_fde,
               native_L_abs_m=native_l, JP_abs_m=abs(jp1-jp0), delta_FDE_G_abs_m=closure)
    return row


def _archived_rows():
    dev = ARCHIVE / "dev" / "PER_WINDOW.csv"
    diag = DIAG_DOC / "PER_WINDOW.csv"
    require(digest(diag) == read(DIAG_DOC / "ARTIFACT_SHA256.json")["PER_WINDOW.csv"],
            "original high-error membership digest differs")
    archived = {(r["fold"], r["window"]): r for r in read_csv(dev)
                if int(r["seed"]) == SEED and r["fold"] in FOLDS}
    high = {(r["fold"], r["window"]) for r in read_csv(diag)
            if r["split"] == DEV and r["is_top10"] == "True" and r["fold"] in FOLDS}
    require(len(archived) == sum(WINDOWS.values()) and high <= archived.keys(), "original score table differs")
    require(all(sum(fold == f for f, _ in high) == math.ceil(WINDOWS[fold]/10) for fold in FOLDS),
            "unexpected original high-error membership")
    return archived, high


def _verify_all_seals(refs, primary, readers):
    proofs = {}
    for fold in FOLDS:
        ref = refs[fold]
        _check_reference(ref, fold)
        root = primary / fold
        path = root / "PREDICTIONS_FROZEN.json"
        seal = read(path)
        protocol_path = root / "EXPORT_CONFIG.json"
        protocol = read(protocol_path)
        require(all(seal[key] == expected for key, expected in
                    dict(fold=fold, purpose=DEV, seed=SEED, windows=WINDOWS[fold], future_reads=0,
                         K=K, export_sha256=digest(protocol_path)).items())
                and protocol["source_hashes"] == ref.source_hashes
                and len(seal["rows"]) == WINDOWS[fold], "unsealed development fold")
        for ordinal, (index, row) in enumerate(zip(ref.indices, seal["rows"]), 1):
            source = _source(ref, index, readers.get(fold))
            expected = _row_identity(ref, index, ordinal, source) | {"export_sha256": seal["export_sha256"]}
            saved, _ = _load_shard(root, expected, source)
            require(saved == row, "per-window shard is outside the frozen manifest")
        proofs[fold] = dict(sha256=digest(path), count=seal["windows"],
                            sealed_unix=seal["sealed_unix"])
    path = primary / "ALL_PREDICTIONS_FROZEN.json"
    if path.exists():
        receipt = read(path)
        require(receipt["folds"] == proofs and receipt["future_reads"] == 0, "global development seal differs")
    else:
        receipt = dict(folds=proofs, future_reads=0, sealed_unix=time.time())
        immutable(path, receipt)
    return receipt


def _selected(row, group):
    if group == "all":
        return True
    if group in ("N1", "N2_8", "N9plus"):
        return row["size_group"] == group
    return {
        "eligible": row["groups_count"] > 1,
        "changed": row["true_change"],
        "top10": row["original_high_error"],
        "single_group": row["groups_count"] == 1,
        "all_singleton_multigroup": row["groups_count"] > 1 and row["paired_groups"] == 0,
        "rename": row["global_relabel"],
        "relative": row["relative"],
    }[group]


def summarize(fold, records):
    """Native per-scene primitives averaged once per window or per agent as prescribed."""
    rows = []
    for group in GROUPS:
        batch = [r for r in records if _selected(r, group)]
        windows, agents = len(batch), sum(r["n"] for r in batch)
        row = dict(fold=fold, group=group, windows=windows, agents=agents)
        for key, attr in (
            ("eligible_windows", "eligible"), ("relative_windows", "relative"),
            ("changed_windows", "changed"), ("global_relabel_windows", "rename"),
            ("all_singleton_multigroup_windows", "all_singleton_multigroup"),
        ):
            row[key] = sum(_selected(r, attr) for r in batch)
        row["effective_change_share"] = row["changed_windows"] / windows if windows else None
        for metric in (*METRICS, "L", "G", "JP", "J_minus_JP"):
            for model in ("S0", "new"):
                if metric in ("minADE", "minFDE"):
                    value = math.fsum(r[f"{model}_agent_{metric}_sum_m"] for r in batch)
                    row[f"{model}_{metric}_m"] = value / agents if agents else None
                else:
                    key = "JP_m" if metric == "JP" else f"{model}_{metric}_m"
                    value = math.fsum(r[key] for r in batch)
                    row[f"{model}_{metric}_m"] = value / windows if windows else None
            row[f"delta_{metric}_cm"] = (100*(row[f"new_{metric}_m"] - row[f"S0_{metric}_m"])
                                          if windows else None)
        for metric in ("JADE", "JFDE"):
            diffs = [r[f"delta_{metric}_m"] for r in batch]
            row.update({
                f"{metric}_improved": sum(x < 0 for x in diffs),
                f"{metric}_worsened": sum(x > 0 for x in diffs),
                f"{metric}_unchanged": sum(x == 0 for x in diffs),
                f"{metric}_median_delta_cm": 100*statistics.median(diffs) if diffs else None,
                f"{metric}_worst_harm_cm": 100*max(0., max(diffs)) if diffs else None,
                f"{metric}_contribution_to_fold_cm": 100*math.fsum(diffs) / len(records),
                f"{metric}_harm_to_fold_cm": 100*math.fsum(max(0., x) for x in diffs) / len(records),
            })
        rows.append(row)
    return rows


def _decision(groups):
    lookup = {(row["fold"], row["group"]): row for row in groups}
    checks = []
    for fold in FOLDS:
        target, count = TARGETS[fold]
        require(lookup[fold, "all"]["windows"] == WINDOWS[fold]
                and lookup[fold, target]["windows"] == count, "decision denominator changed")
        requested = [("all", "JADE", TOL), ("all", "JFDE", TOL)]
        requested += ([(target, "JADE", TOL), (target, "JFDE", TOL)] if fold == "hotel"
                      else [(target, "JFDE", -.01)])
        for group, metric, limit in requested:
            row = lookup[fold, group]
            value = row[f"new_{metric}_m"] - row[f"S0_{metric}_m"]
            checks.append(dict(fold=fold, group=group, metric=metric, windows=row["windows"],
                               delta_m=value, max_delta_m=limit, passed=value <= limit))
    fold_pass = {fold: all(row["passed"] for row in checks if row["fold"] == fold) for fold in FOLDS}
    return dict(checks=checks, fold_pass=fold_pass, exact_marginals=True,
                supports_univ=fold_pass["univ"], supports_both=all(fold_pass.values()),
                evaluated_test=False)


def _score_fold(ref, root, read_path, prior, high):
    records = []
    reader = FormalData(ref.split, phase="scoring", verify=False)
    export_sha = digest(root / "EXPORT_CONFIG.json")
    for ordinal, index in enumerate(ref.indices, 1):
        source = _source(ref, index, read_path)
        expected = _row_identity(ref, index, ordinal, source) | {"export_sha256": export_sha}
        seal, tensors = _load_shard(root, expected, source)
        record = ref.split["records"][index]
        require(digest(record["path"]) == record["sha256"], "development truth changed after inference")
        batch = reader.batch([index], DEV, truth=True)
        n = expected["n"]
        require(batch.valid.shape == (1, n) and bool(batch.valid.all())
                and bool(torch.isfinite(batch.future).all()), "invalid packed development truth")
        pairs = greedy_observed_pairs(batch.observed[0].numpy().astype(np.float64))
        require(np.array_equal(pairs, tensors["pairs"]), "sealed development groups differ")
        scores = _score_pair(source, tensors["paths"], batch.future, batch.valid, pairs)
        key = ref.fold, expected["window"]
        original = prior[key]
        original_diff = max(abs(scores[f"S0_{metric}_m"] - float(original[f"S0_{metric}"]))
                            for metric in (*METRICS, "L", "G"))
        require(int(original["n"]) == n and original_diff <= TOL, "native S0 archived metrics failed replay")
        if not seal["true_change"]:
            require(all(scores[f"delta_{metric}_m"] == 0 for metric in METRICS),
                    "unchanged joint ensemble changed native scores")
        flags = {name: seal[name] for name in (
            "groups_count", "paired_groups", "identity", "relative", "global_relabel", "true_change")}
        records.append(dict(fold=ref.fold, split=DEV, seed=SEED, window=expected["window"], n=n,
                            size_group=group_for_agents(n), original_high_error=key in high,
                            S0_replay_max_abs_m=original_diff, **flags, **scores))
    return records


@torch.no_grad()
def score_all(*, refs: dict, output: Path, doc: Path, read_paths: dict | None = None,
              runtime: dict | None = None):
    """Verify every pair across both folds before the first development target is loaded."""
    started = time.monotonic()
    output, doc = Path(output), Path(doc)
    primary = output / "development"
    readers = read_paths or {}
    require(set(refs) == set(FOLDS), "both original development folds are required")
    seal = _verify_all_seals(refs, primary, readers)
    prior, high = _archived_rows()
    first_truth = time.time()
    require(first_truth >= seal["sealed_unix"], "truth was opened before the global prediction seal")
    records, groups = [], []
    for fold in FOLDS:
        scored = _score_fold(refs[fold], primary / fold, readers.get(fold), prior, high)
        records.extend(scored)
        groups.extend(summarize(fold, scored))
    doc.mkdir(parents=True, exist_ok=True)
    csvwrite(doc / "GROUPS.csv", groups)
    csvwrite(doc / "PER_WINDOW.csv", records)
    decision = _decision(groups)
    receipt = dict(
        status="COMPLETE", purpose=DEV, seed=SEED, K=K, windows=len(records),
        all_folds_frozen_unix=seal["sealed_unix"], first_development_future_read_unix=first_truth,
        prediction_seal_sha256=digest(primary / "ALL_PREDICTIONS_FROZEN.json"),
        inverse_bitwise_windows=len(records), determination=decision,
        native_max_abs_m={k: max(row[k] for row in records) for k in
                          ("native_minADE_max_abs_m", "native_minFDE_max_abs_m", "native_L_abs_m",
                           "JP_abs_m", "delta_FDE_G_abs_m", "S0_replay_max_abs_m")},
        score_seconds=time.monotonic()-started, runtime=public_metadata(runtime or {}),
        output_sha256={name: digest(doc / name) for name in ("GROUPS.csv", "PER_WINDOW.csv")},
    )
    write(primary / "SCORE_RECEIPT.json", receipt)
    write(doc / "DEVELOPMENT_SCORE.json", receipt)
    build_report(doc, groups, receipt)
    return receipt | {"groups": groups}


def _number(value, fmt):
    return "N/A" if value is None else format(value, fmt)


def build_report(doc, groups, receipt):
    lookup = {(row["fold"], row["group"]): row for row in groups}
    hotel, univ = (lookup[fold, TARGETS[fold][0]] for fold in FOLDS)
    decision = receipt["determination"]
    all_rows = [lookup[fold, "all"] for fold in FOLDS]
    harm_parts = []
    for total in all_rows:
        fold = total["fold"]
        if total["JFDE_worsened"]:
            largest = max((lookup[fold, size] for size in ("N1", "N2_8", "N9plus")),
                          key=lambda row: row["JFDE_harm_to_fold_cm"])
            harm_parts.append(f"{fold.upper()} {total['JFDE_worsened']} 窗 JFDE 上升，"
                              f"累积恶化以 {GROUP_LABEL[largest['group']]} 组最多")
    harms = "，".join(harm_parts)
    if not harms:
        harms = "两折逐窗 JFDE 均未上升"
    lead = (
        f"在完整 K20 下，HOTEL 全部 440 个 2–8 人窗和 UNIV 全部 61 个密集窗的 JFDE"
        f"分别变化 **{hotel['delta_JFDE_cm']:+.4f} cm**、**{univ['delta_JFDE_cm']:+.4f} cm**，"
        f"{harms}，逐人 minADE、minFDE **完全保留**。"
    )
    lines = [
        "# HOTEL 与 UNIV 的局部行人组重新组合", "", lead, "",
        "原开发集单种子 2035。HOTEL 使用冻结 S24，UNIV 使用冻结 S16。"
        "两折均采用各自原 train 训练的第 12 轮匹配器，搬运原生 K20 的完整 12 帧轨迹。",
        "", "## 全部指标与分组", "",
        "表中每格为 S0 → 重组（m），括号为配对差（cm）。"
        "负差表示误差下降，联合项按窗等权，边缘项按有效人次加权。",
        "",
        "| 折 | 分组 | 窗／人次 | JADE | JFDE | minADE | minFDE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in groups:
        vals = []
        for metric in METRICS:
            base, new = (row[f"{model}_{metric}_m"] for model in ("S0", "new"))
            vals.append(f"{base:.6f} → {new:.6f} ({row[f'delta_{metric}_cm']:+.4f})" if base is not None else "N/A")
        lines.append("| " + " | ".join([row["fold"].upper(), GROUP_LABEL[row["group"]],
                                       f"{row['windows']}／{row['agents']}", *vals]) + " |")
    lines.extend([
        "", "## 哪些窗改变，哪些窗承担代价", "",
        "两组或以上可预测相对换槽。只有整场候选多重集实际改变才计有效改变。"
        "所有组共享一个非身份置换时计为整场换名。高误差窗用原冻结诊断的 S0 前 10% 成员。",
        "",
        "| 折 | 分组 | 相对换槽／有效改变／整场换名 | 指标 | 降／升／平窗数 | 中位配对差（cm） | 最大恶化（cm） | 对全折贡献（cm） |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ])
    for row in groups:
        for metric in ("JADE", "JFDE"):
            lines.append("| " + " | ".join([
                row["fold"].upper(), GROUP_LABEL[row["group"]],
                f"{row['relative_windows']}／{row['changed_windows']}／{row['global_relabel_windows']}", metric,
                "／".join(str(row[f"{metric}_{suffix}"]) for suffix in ("improved", "worsened", "unchanged")),
                _number(row[f"{metric}_median_delta_cm"], "+.4f"),
                _number(row[f"{metric}_worst_harm_cm"], ".4f"),
                _number(row[f"{metric}_contribution_to_fold_cm"], "+.4f"),
            ]) + " |")
    lines.extend([
        "", "有效换槽比例为 " + "、".join(f"{r['fold'].upper()} {r['effective_change_share']:.2%}"
                                        for r in all_rows) + "，分母为各折全体开发窗。",
        "分组对全折的贡献为该组全部逐窗配对差之和除以该折总窗数。"
        "有效改变窗与人数分组可重叠，全折判断使用完整分母。",
        "", "## 同窗真值松弛空间", "",
        "J−Jₚ 是固定组划分后分别按真值为每组选槽可减少的误差，只用于量化诊断空间。",
        "",
        "| 折 | 分组 | 窗数 | S0 L／G（m） | S0 Jₚ（m） | S0 J−Jₚ（cm） | 重组 J−Jₚ（cm） |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in groups:
        base = _number(row["S0_J_minus_JP_m"]*100 if row["windows"] else None, ".4f")
        new = _number(row["new_J_minus_JP_m"]*100 if row["windows"] else None, ".4f")
        lines.append(f"| {row['fold'].upper()} | {GROUP_LABEL[row['group']]} | {row['windows']} | "
                     f"{_number(row['S0_L_m'], '.6f')}／{_number(row['S0_G_m'], '.6f')} | "
                     f"{_number(row['S0_JP_m'], '.6f')} | {base} | {new} |")
    lines.extend([
        "", "## 固定判断", "",
        "| 折 | 分组 | 指标 | 窗数 | 实际配对差（cm） | 最大允许差（cm） | 通过 |",
        "|---|---|---|---:|---:|---:|---|",
    ])
    for check in decision["checks"]:
        lines.append(f"| {check['fold'].upper()} | {GROUP_LABEL[check['group']]} | {check['metric']} | "
                     f"{check['windows']} | {100*check['delta_m']:+.6f} | "
                     f"{100*check['max_delta_m']:+.6f} | {'是' if check['passed'] else '否'} |")
    if decision["supports_both"]:
        conclusion = "UNIV 密集组达到 1 cm 工程尺度，两折保护项与逐人恒等核验全部通过，支持继续。"
    elif decision["supports_univ"]:
        conclusion = "UNIV 达到密集组工程尺度并守住全体联合指标，本次数据仅支持 UNIV。"
    elif all(row["changed_windows"] == 0 for row in all_rows):
        conclusion = "本次匹配器未学到能实际改变整场候选的相对排列，结束这一版。"
    elif 0 > univ["delta_JFDE_cm"] > -1:
        conclusion = (f"UNIV 密集组只有 {abs(univ['delta_JFDE_cm'])*10:.4f} mm 的平均 JFDE 改善，"
                      "未达到 1 cm 工程尺度，结束这一版。")
    else:
        conclusion = "UNIV 密集组及联合保护指标未同时达到继续条件，结束这一版。"
    lines.extend(["", conclusion, "", "## 封存与核验", "",
                  f"{receipt['inverse_bitwise_windows']} 窗在读取开发未来前封存预测与置换。"
                  "逐人完整轨迹逆置换逐元素位级相等，单组输出整场直接复制。", ""])
    floats = receipt["native_max_abs_m"]
    lines.append(
        "原生浮点最大差为 " + "，".join(
            f"{label} {floats[key]:.3g} m" for key, label in (
                ("native_minADE_max_abs_m", "逐人 minADE"), ("native_minFDE_max_abs_m", "逐人 minFDE"),
                ("native_L_abs_m", "每窗 L"), ("JP_abs_m", "同一划分的 Jₚ"),
                ("delta_FDE_G_abs_m", "ΔJFDE−ΔG"), ("S0_replay_max_abs_m", "S0 归档回放")))
        + "。"
    )
    lines.extend(["", f"封存校验和评分耗时 {receipt['score_seconds']:.2f} 秒。",
                  "运行配置、缓存与续训记录见目录内收据。", "",
                  "[分组 CSV](GROUPS.csv) · [匿名逐窗 CSV](PER_WINDOW.csv) · [评分收据](DEVELOPMENT_SCORE.json)", ""])
    (Path(doc) / "README.md").write_text("\n".join(lines), encoding="utf-8")
