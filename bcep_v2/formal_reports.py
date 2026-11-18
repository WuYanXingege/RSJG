"""Publish only small, relative-path, dev/frozen result tables from a report checkout."""
import csv
import math
import statistics
import subprocess
import time
from pathlib import Path

from bcep.contracts import canonical_hash, digest, require
from bcep_joint.common import read, write
from .formal_manifest import DIAGNOSTIC, FOLDS, METRIC, OFFICIAL, immutable, paired_indexes, write_csv
from .frozen_test import case_folder
from .workflow import SEEDS

DOCS = Path("docs/research/bcep_v2/formal_fivefold_top5_20261118")
PUBLIC_BRANCH = "report/bcep-ebjd-v2-formal-top5-20261118"
TRAJECTORY = ("minADE", "minFDE", "JADE", "JFDE", "scene_weighted_minADE", "scene_weighted_minFDE",
              "coordination_gap_ADE", "coordination_gap_FDE")


def fold_tables(root, fold, proofs):
    dest = root / "public" / fold
    for stage in METRIC:
        proof = proofs[stage]
        history = read(root / fold / stage / "EPOCH_METRICS.json")
        write_csv(dest / stage / "EPOCH_METRICS.csv",
                  [dict(fold=fold, stage=stage, selection_metric=METRIC[stage], **r) for r in history])
        write_csv(dest / stage / "DEV_TOP5.csv", proof["ranks"])
    for name, rows in paired_indexes(proofs).items():
        immutable(dest / (name + ".json"), rows)
        write_csv(dest / (name + ".csv"), rows)
    lines = [f"# {fold.upper()} 两大阶段训练结果", "",
             "| 子流程 | 完整轮次 | 更新 | 最佳轮 | dev 值 | 停止原因 | 训练小时 | 验证小时 |",
             "|---|---:|---:|---:|---:|---|---:|---:|"]
    for s in METRIC:
        p = proofs[s]
        h = read(root / fold / s / "EPOCH_METRICS.json")
        lines.append(f"| {s} | {p['epoch']} | {p['update']} | {p['selected']['epoch']} | "
                     f"{p['selected']['value']:.7g} | {p['stop_reason']} | "
                     f"{sum(r['train_seconds'] for r in h)/3600:.3f} | "
                     f"{sum(r['validation_seconds'] for r in h)/3600:.3f} |")
    lines += ["", "Stage 1 的基础候选及精修 train/dev 候选均通过成员和内容哈希核验。"
              "Stage 2 使用固定的 Provider dev rank1。各子流程的训练和验证时间单列，保存开销另计。"]
    (dest / "REPORT_ZH.md").write_text("\n".join(lines) + "\n")


def rows_from(directory):
    path = directory / "PER_SEED_METRICS.csv"
    require(digest(path) == read(directory / "COMPLETE.json")["metrics_sha256"], "score digest differs")
    with path.open() as f:
        return list(csv.DictReader(f))


def annotated(row, selected, frozen, bank, steps):
    keys = ("fold", "stage", "macro_stage", "dev_rank", "epoch", "update", "dev_value", "checkpoint_sha256",
            "sources", "config", "split", "members", "upstream", "density_checkpoint")
    meta = {k: selected[k] for k in keys}
    meta.update(head_checkpoint=selected.get("head_checkpoint", selected["checkpoint_sha256"]),
                K=20, steps=steps, role=OFFICIAL if selected["dev_rank"] == 1 else DIAGNOSTIC,
                freeze=frozen["content_sha256"], bank_sha256=bank)
    return dict(row, **meta, seed=int(row["seed"]))


def summary(rows, metrics):
    require(sorted(r["seed"] for r in rows) == list(SEEDS), "five different inference seeds required")
    first = rows[0]
    result = {k: first[k] for k in ("fold", "stage", "macro_stage", "dev_rank", "epoch", "update",
                                     "checkpoint_sha256", "density_checkpoint", "head_checkpoint", "role",
                                     "group", "method", "K", "steps", "freeze", "split", "members", "sources", "config")}
    result["seeds"] = list(SEEDS)
    result["upstream"] = first["upstream"]
    result["banks_by_seed"] = {r["seed"]: r["bank_sha256"] for r in rows}
    for m in metrics:
        values = [float(r[m]) for r in rows]
        result[m + "_mean"] = statistics.mean(values)
        result[m + "_sd"] = statistics.stdev(values)
    return result


def official_macro(official):
    result = []
    for method in ("G", "H"):
        subset = [r for r in official if r["group"] == "all" and r["method"] == method]
        require(len(subset) == len(FOLDS)*len(SEEDS), "incomplete official matrix")
        macro = []
        for seed in SEEDS:
            r = [x for x in subset if x["seed"] == seed]
            require({x["fold"] for x in r} == set(FOLDS), "missing fold")
            row = dict(r[0], fold="FIVE_FOLD_MACRO",
                       bank_sha256={x["fold"]: x["bank_sha256"] for x in r})
            for key in ("epoch", "update", "checkpoint_sha256", "head_checkpoint", "density_checkpoint",
                        "split", "members", "upstream"):
                row[key] = {x["fold"]: x[key] for x in r}
            row.update({metric: statistics.mean(float(x[metric]) for x in r) for metric in TRAJECTORY})
            macro.append(row)
        result.append(summary(macro, TRAJECTORY))
    return result


def final_tables(root, frozen):
    official, joint, provider, joint_avg, head_avg, off_avg, diagnostics = [], [], [], [], [], [], []
    for fold in FOLDS:
        fr = frozen["folds"][fold]
        rank1_dir = case_folder(root, fr["JOINT_TOP5"][0])
        refbanks = read(rank1_dir / "IDENTITY.json")["banks"]
        ref_fde = {}
        for ck in fr["JOINT_TOP5"]:
            data = [annotated(r, ck, frozen, refbanks[r["seed"]], 20)
                    for r in rows_from(case_folder(root, ck)) if r["method"] in ("G", "H")]
            for group, method in sorted({(r["group"], r["method"]) for r in data}):
                rows = [r for r in data if r["group"] == group and r["method"] == method]
                avg = summary(rows, TRAJECTORY)
                if ck["dev_rank"] == 1:
                    official.extend(rows)
                    off_avg.append(avg)
                if method == "H":
                    joint.extend(rows)
                    joint_avg.append(avg)
                    for row in rows:
                        key = (group, row["seed"])
                        v = float(row["minFDE"])
                        if ck["dev_rank"] == 1:
                            ref_fde[key] = v
                        else:
                            require(math.isclose(v, ref_fde[key], abs_tol=1e-6, rel_tol=1e-6), "fixed K20 minFDE")
        head_rows = rows_from(root / fold / "evaluation/head_matrix")
        for ck in fr["PROVIDER_TOP5"]:
            data = [annotated(r, ck, frozen, r["bank_sha256"], 0) for r in head_rows
                    if int(r["dev_rank"]) == ck["dev_rank"]]
            provider.extend(data)
            for g in sorted({r["group"] for r in data}):
                rows = [r for r in data if r["group"] == g]
                head_avg.append(summary(rows, ["minFDE"]))
                if ck["dev_rank"] == 1:
                    for r in rows:
                        require(math.isclose(float(r["minFDE"]), ref_fde[g, r["seed"]], abs_tol=1e-6, rel_tol=1e-6),
                                "provider vs hard-endpoint joint minFDE differs")
        for items, metric, label in [(joint_avg, "JADE", "TEST_POSTHOC_BEST_H_JADE_WITHIN_DEV_TOP5"),
                                      (head_avg, "minFDE", "TEST_POSTHOC_BEST_HEAD_MINFDE_WITHIN_DEV_TOP5")]:
            candidates = [r for r in items if r["fold"] == fold and r["group"] == "all"]
            require(sorted(r["dev_rank"] for r in candidates) == [1, 2, 3, 4, 5], "incomplete diagnostic")
            winner = min(candidates, key=lambda r: (r[metric+"_mean"], r["dev_rank"], r["epoch"]))
            diagnostics.append(dict(winner, role=label))
    dest = root / "public"
    macros = official_macro(official)
    for name, rows in [("OFFICIAL_PER_SEED", official), ("OFFICIAL_FIVEFOLD_SUMMARY", off_avg + macros),
                       ("JOINT_TOP5_TEST_PER_SEED", joint), ("JOINT_TOP5_TEST_SUMMARY", joint_avg),
                       ("HEAD_TOP5_TEST_PER_SEED", provider), ("HEAD_TOP5_TEST_SUMMARY", head_avg),
                       ("SELECTION_DIAGNOSTICS", diagnostics)]:
        write_csv(dest / (name + ".csv"), rows)
    lines = ["# BCEP–EBJD v2 五折测试", "",
             "五折全部冻结。主表使用每折官方 dev rank1，单位为米。各推理 seed 先取五折等权平均，再计算五 seed 的均值及样本标准差。",
             "", "| 方法 | minADE | minFDE | JADE | JFDE |", "|---|---:|---:|---:|---:|"]
    for r in macros:
        scores = [f"{r[m+'_mean']:.5f} ± {r[m+'_sd']:.5f}" for m in TRAJECTORY[:4]]
        lines.append("| " + " | ".join([r["method"], *scores]) + " |")
    lines += ["", "minADE/minFDE 按有效行人出现次数加权，JADE/JFDE 按场景等权。G 使用原生候选顺序，H 使用学习配对。",
              "", "固定 K20 的 scene_weighted_minFDE 是允许用真值重排时的联合 FDE 下界。"
              "coordination_gap_FDE 是 H-JFDE 与该下界之差。对应 ADE 量仅描述已经生成的路径集合。",
              "", "诊断表按每个 checkpoint 的五 seed 均值选探索赢家，同分按 dev 排名及 epoch。"
              "推理种子标准差反映采样波动。各折只有一次训练；测试后挑选的结果含选择偏差。",
              "", "项目早期开发曾参考既有 ZARA2 测试诊断。本轮配置、训练规则及 Top5 矩阵在新测试前冻结。"]
    (dest / "REPORT_ZH.md").write_text("\n".join(lines) + "\n")


def publish(root, checkout, label):
    """A report transport failure is recorded separately from the scientific state."""
    import shutil
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=checkout, text=True, stderr=subprocess.STDOUT,
                                       timeout=90).strip()
    try:
        require(git("branch", "--show-current") == PUBLIC_BRANCH, "wrong reporting branch")
        files = sorted((root / "public").rglob("*"))
        allowed = {"PROVIDER_TOP5.json", "JOINT_TOP5.json", "PROVIDER_TOP5.csv", "JOINT_TOP5.csv",
                   "DEV_TOP5.csv", "EPOCH_METRICS.csv", "REPORT_ZH.md", "OFFICIAL_PER_SEED.csv",
                   "OFFICIAL_FIVEFOLD_SUMMARY.csv", "JOINT_TOP5_TEST_PER_SEED.csv", "JOINT_TOP5_TEST_SUMMARY.csv",
                   "HEAD_TOP5_TEST_PER_SEED.csv", "HEAD_TOP5_TEST_SUMMARY.csv", "SELECTION_DIAGNOSTICS.csv"}
        paths = []
        for src in files:
            if src.is_dir():
                continue
            require(src.name in allowed and not src.is_symlink(), "unrecognized public artifact")
            body = src.read_text()
            require("/media/" not in body and "/home/" not in body, "private path in public artifact")
            dst = DOCS / src.relative_to(root / "public")
            (checkout / dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, checkout / dst)
            paths.append(str(dst))
        if paths:
            git("add", "--", *paths)
            if git("diff", "--cached", "--name-only", "--", *paths):
                git("commit", "--only", "-m", "Record v2 formal results " + label, "--", *paths)
        git("push", "origin", "HEAD:refs/heads/" + PUBLIC_BRANCH)
        remote = git("remote", "get-url", "--push", "origin")
        sha = git("rev-parse", "HEAD")
        require(git("ls-remote", "--exit-code", remote, "refs/heads/" + PUBLIC_BRANCH).split()[0] == sha,
                "remote report SHA differs")
        write(root / "publishing" / (label + ".json"), dict(status="PUBLISHED", sha=sha, unix=time.time()))
    except Exception as exc:
        write(root / "publishing" / (label + ".json"), dict(status="PUBLICATION_PENDING", reason=str(exc)))
