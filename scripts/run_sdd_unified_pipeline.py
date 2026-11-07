#!/usr/bin/env python3
"""Resumable JMM TrajNet-SDD asset preparation and unified EBJD pipeline."""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, shutil, subprocess, tempfile, time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

PAPER = Path(__file__).resolve().parents[1]
PYTHON = Path("/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/.conda/rsjg/bin/python")
RECIPE = PAPER / "configs/asset_preparation/sdd_jmm_gdts.yaml"
BASE_CONFIG = PAPER / "configs/resolved/zara1.json"
SEEDS = tuple(range(2035, 2040))
INPUT_ACCEPTANCE = Path(
    "/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/"
    "RSJG_EBJD_assets/experiments/"
    "ebjd_input_pipeline_applicability_20261106_221435")

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""): digest.update(block)
    return digest.hexdigest()

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True); handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)

class Pipeline:
    MINIMUM_STAGE_FREE_BYTES = 32 * 1024 ** 3

    def __init__(
            self, root, commit, migration_reason=None,
            migration_optimizer_updates=None):
        self.root, self.commit = Path(root).resolve(), commit
        self.runtime = self.root / "04_matched_gdts/runtime"
        self.cache = self.root / "04_matched_gdts/cache"
        self.independent_cache = (
            self.cache / "sdd/sdd/data_batches_zstd_v1")
        self.joint_cache = (
            self.cache / "sdd/sdd/data_batches_joint_v2_zstd_v1")
        data = Path("/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_assets/data/sdd_trajnet_jmm/manifests")
        self.train, self.valid = data / "train/train_manifest.json", data / "validation/validation_manifest.json"
        for name in ("00_identity","01_map_recovery","02_data_manifests","03_goal_pretrain","04_matched_gdts","05_proposal_banks","06_ebjd_training","07_evaluation","logs","reports","receipts"):
            (self.root/name).mkdir(parents=True, exist_ok=True)
        self.state_path, self.budget_path = self.root/"PIPELINE_STATE.json", self.root/"GPU_BUDGET.json"
        if not self.budget_path.exists(): atomic_json(self.budget_path, {"limit_seconds":604800.0,"consumed_seconds":0.0,"stages":[]})
        run_manifest = self.root/"RUN_MANIFEST.json"
        if not run_manifest.exists():
            atomic_json(run_manifest, {
                "schema":"ebjd-sdd-jmm-unified-run-v1",
                "fold":"sdd_trajnet_jmm", "training_seed":3101,
                "source_commit":self.commit, "epochs":30, "worlds":20,
                "ddim_steps":20, "rollout":False, "projection":False,
                "architecture":"192/4/768, 6 FJAB, FULL/CTF/Raw",
                "experiment_root":str(self.root)})
        else:
            identity=json.loads(run_manifest.read_text())
            if identity.get("source_commit") != self.commit:
                identity.setdefault("execution_migrations",[]).append({
                    "from":identity.get("source_commit"), "to":self.commit,
                    "reason": migration_reason or
                    "stage-boundary resumable execution handoff",
                    "optimizer_updates_before_migration":
                    migration_optimizer_updates})
                identity["source_commit"]=self.commit
                atomic_json(run_manifest,identity)
        for filename, schema in (("CONFIG_HASHES.json","ebjd-sdd-config-hashes-v1"),
                                 ("CHECKPOINT_MANIFEST.json","ebjd-sdd-checkpoints-v1")):
            path=self.root/filename
            if not path.exists(): atomic_json(path,{"schema":schema,"status":"PENDING"})

    def state(self, stage, status, **extra):
        atomic_json(self.state_path, {"schema":"ebjd-sdd-jmm-unified-pipeline-v1","stage":stage,"status":status,"source_commit":self.commit,"experiment_root":str(self.root),"updated_at_utc":datetime.now(timezone.utc).isoformat(),**extra})

    def run(self, stage, command):
        self.require_free_space(stage)
        log = self.root/"logs"/(stage.lower()+".log"); self.state(stage,"RUNNING",command=command,log=str(log))
        started=time.perf_counter(); outcome="failed"
        try:
            environment = os.environ.copy()
            if stage.startswith("CACHE_"):
                environment.update({
                    "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                    "OPENBLAS_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2",
                })
            with log.open("a",encoding="utf-8") as handle:
                result=subprocess.run(
                    command, cwd=PAPER, stdout=handle,
                    stderr=subprocess.STDOUT, check=False, env=environment)
            if result.returncode: raise RuntimeError(f"{stage} exited {result.returncode}")
            outcome="complete"
        except BaseException as error:
            self.state(
                stage, "FAILED", failed_stage=stage, resume_from=stage,
                reason=f"{type(error).__name__}: {error}", log=str(log))
            raise
        finally:
            budget=json.loads(self.budget_path.read_text()); seconds=time.perf_counter()-started
            budget["consumed_seconds"]+=seconds; budget["stages"].append({"stage":stage,"seconds":seconds,"status":outcome}); atomic_json(self.budget_path,budget)

    def require_free_space(self, stage):
        free = shutil.disk_usage(self.root).free
        if free < self.MINIMUM_STAGE_FREE_BYTES:
            self.state(
                stage, "PAUSED_NO_SPACE", failed_stage=stage,
                resume_from=stage, free_bytes=free,
                required_free_bytes=self.MINIMUM_STAGE_FREE_BYTES,
                reason="insufficient filesystem headroom before stage start")
            raise RuntimeError(
                f"{stage} requires at least {self.MINIMUM_STAGE_FREE_BYTES} "
                f"free bytes; found {free}")

    def verify(self):
        train,valid=json.loads(self.train.read_text()),json.loads(self.valid.read_text())
        actual=((train["statistics"]["scenes"],train["statistics"]["agents"]),(valid["statistics"]["scenes"],valid["statistics"]["agents"]))
        if actual!=((5720,8940),(1999,2829)): raise RuntimeError(f"protocol counts: {actual}")
        if "nexus_2" not in {x["sequence"] for x in train["source_files"]}: raise RuntimeError("nexus_2 absent")
        atomic_json(self.root/"02_data_manifests/DATA_PROTOCOL_RECEIPT.json",{"train":str(self.train),"train_sha256":sha256(self.train),"validation":str(self.valid),"validation_sha256":sha256(self.valid),"train_statistics":train["statistics"],"validation_statistics":valid["statistics"]})

    def wait_gpu(self):
        while subprocess.run([str(PYTHON),"-c","import torch;raise SystemExit(0 if torch.cuda.is_available() else 1)"],check=False).returncode:
            self.state("WAIT_GPU","WAITING_FOR_GPU",retry_seconds=60); time.sleep(60)

    def caches(self):
        self.require_free_space("CACHE_INDEPENDENT")
        common=[str(PYTHON),"scripts/run_legacy_gdts_stage.py","--recipe",str(RECIPE),"--runtime-root",str(self.runtime),"--batch-cache-root",str(self.cache)]
        if not (self.independent_cache/"finished_train_batches.txt").exists(): self.run("CACHE_INDEPENDENT",common+["--stage","scene_cache_independent"])
        self.require_free_space("CACHE_JOINT")
        bound_common=[str(PYTHON),"scripts/build_manifest_bound_gdts_cache.py","--recipe",str(RECIPE),"--runtime-root",str(self.runtime),"--batch-cache-root",str(self.cache)]
        if not (self.joint_cache/"finished_train_batches.txt").exists(): self.run("CACHE_JOINT_TRAIN",bound_common+["--manifest",str(self.train),"--split","train"])
        if not (self.joint_cache/"finished_valid_batches.txt").exists(): self.run("CACHE_JOINT_VALIDATION",bound_common+["--manifest",str(self.valid),"--split","validation"])
        expected={"train":5720,"valid":1999,"test":1999}
        for split,count in expected.items():
            marker=self.joint_cache/f"finished_{split}_batches.txt"
            if not marker.exists() or str(count) not in marker.read_text():
                raise RuntimeError(
                    f"manifest-bound joint cache mismatch: {split} != {count}")
        for split,manifest in (("train",self.train),("valid",self.valid)):
            receipt=self.root/f"receipts/CACHE_PARITY_{split.upper()}.json"
            if not receipt.exists(): self.run("AUDIT_CACHE_"+split.upper(),[str(PYTHON),"scripts/audit_manifest_gdts_cache.py","--recipe",str(RECIPE),"--manifest",str(manifest),"--runtime-root",str(self.runtime),"--cache-dir",str(self.joint_cache/f"{split}_batches"),"--output",str(receipt)])

    def gdts(self):
        common=[str(PYTHON),"scripts/run_legacy_gdts_stage.py","--recipe",str(RECIPE),"--runtime-root",str(self.runtime),"--batch-cache-root",str(self.cache)]
        goal=self.runtime/"output/sdd/runs/sdd_jmm_goal_pretrain_seed3101_v1/saved_models/goal_pretrain_best_model.pt"
        goal_last=self.runtime/"output/sdd/runs/sdd_jmm_goal_pretrain_seed3101_v1/saved_models/goal_pretrain_last_model.pt"
        goal_receipt=self.root/"03_goal_pretrain/GOAL_PRETRAIN_COMPLETION.json"
        if goal_receipt.exists():
            receipt=json.loads(goal_receipt.read_text(encoding="utf-8"))
            if (receipt.get("stage_status") != "COMPLETED" or
                    Path(receipt.get("selected_checkpoint", "")).resolve() != goal.resolve() or
                    not goal.exists() or receipt.get("selected_checkpoint_sha256") != sha256(goal)):
                raise RuntimeError("invalid Goal-pretrain completion receipt")
        else:
            completed_naturally=False
            if goal.exists() and goal_last.exists():
                import torch
                last_payload=torch.load(goal_last,map_location="cpu",weights_only=False)
                completed_naturally=(
                    int(last_payload.get("epoch",0)) >=
                    int(yaml.safe_load(RECIPE.read_text())["goal_pretrain"]["epochs"]))
            if not completed_naturally:
                self.run("GOAL_PRETRAIN",common+["--stage","goal"])
            if not goal.exists() or not goal_last.exists():
                raise RuntimeError("goal selected or last checkpoint absent")
            import torch
            best_payload=torch.load(goal,map_location="cpu",weights_only=False)
            last_payload=torch.load(goal_last,map_location="cpu",weights_only=False)
            atomic_json(goal_receipt,{
                "stage_status":"COMPLETED",
                "completion_reason":"MAX_EPOCHS",
                "planned_epochs":int(best_payload["training_epochs_planned"]),
                "completed_epochs":int(last_payload["epoch"]),
                "selected_checkpoint":str(goal),
                "selected_checkpoint_sha256":sha256(goal),
                "selected_epoch":int(best_payload["epoch"]),
                "validation_goal_BCE":float(best_payload["best_metrics"]["goal_BCE"]),
            })
        if not goal.exists(): raise RuntimeError("goal selected checkpoint absent")
        base=self.runtime/"output/sdd/runs/sdd_jmm_matched_gdts_seed3101_v1/saved_models/best_model.pt"
        base_receipt=self.root/"04_matched_gdts/MATCHED_GDTS_COMPLETION.json"
        if not base_receipt.exists():
            self.run("MATCHED_GDTS",common+["--stage","baseline","--goal-checkpoint",str(goal)])
        if not base.exists() or not base_receipt.exists():
            raise RuntimeError("matched GDTS selected checkpoint or completion receipt absent")
        receipt=json.loads(base_receipt.read_text(encoding="utf-8"))
        if (receipt.get("stage_status") != "COMPLETED" or
                Path(receipt.get("selected_checkpoint", "")).resolve() != base.resolve() or
                receipt.get("selected_checkpoint_sha256") != sha256(base) or
                receipt.get("goal_pretrain_initialization_sha256") != sha256(goal)):
            raise RuntimeError("invalid matched-GDTS completion receipt")
        return goal,base

    def assets(self, base):
        digest=sha256(base); result=self.root/"04_matched_gdts/evaluation/results.json"
        if not result.exists(): self.run("MATCHED_GDTS_EVAL",[str(PYTHON),"scripts/evaluate_matched_gdts.py","--recipe",str(RECIPE),"--runtime-root",str(self.runtime),"--scene-cache-root",str(self.cache),"--checkpoint",str(base),"--checkpoint-sha256",digest,"--manifest",str(self.valid),"--parity-receipt",str(self.root/"receipts/CACHE_PARITY_VALID.json"),"--output-dir",str(result.parent),"--split","valid","--seeds",*map(str,SEEDS)])
        self.gdts_eval_consistency(base, digest, result)
        train_dir=self.root/"05_proposal_banks/train"; train_bank=train_dir/"gdts_proposal_train_seed3101.npz"
        if not train_bank.exists(): self.run("EXPORT_TRAIN_BANK",[str(PYTHON),"scripts/export_zara1_train_proposals.py","--recipe",str(RECIPE),"--checkpoint",str(base),"--checkpoint-sha256",digest,"--manifest",str(self.train),"--manifest-sha256",sha256(self.train),"--scene-cache-root",str(self.cache),"--runtime-root",str(self.runtime),"--output-dir",str(train_dir),"--seed","3101"])
        pattern=self.root/"05_proposal_banks/validation/seed_{seed}/gdts_proposal_B_seed{seed}.npz"
        for seed in SEEDS:
            target=Path(str(pattern).format(seed=seed))
            if not target.exists(): self.run(f"EXPORT_VALIDATION_BANK_{seed}",[str(PYTHON),"scripts/export_zara1_endpoint_banks.py","gdts","--recipe",str(RECIPE),"--runtime-root",str(self.runtime),"--scene-cache-root",str(self.cache),"--checkpoint",str(base),"--checkpoint-sha256",digest,"--manifest",str(self.valid),"--manifest-sha256",sha256(self.valid),"--output-dir",str(target.parent),"--seed",str(seed)])
        return train_bank,str(pattern),digest,result

    def gdts_eval_consistency(self, base, digest, formal_result):
        """Freeze the SDD baseline protocol before creating proposal banks."""
        receipt_path=(
            self.root/"receipts/SDD_GDTS_EVAL_CONSISTENCY.json")
        receipt=(json.loads(receipt_path.read_text(encoding="utf-8"))
                 if receipt_path.exists() else {})
        if receipt.get("status") == "COMPLETE":
            summary_path=Path(receipt["summary"])
            protocol_path=Path(receipt["frozen_protocol"])
            if (not summary_path.exists() or not protocol_path.exists() or
                    receipt.get("checkpoint_sha256") != digest or
                    receipt.get("manifest_sha256") != sha256(self.valid)):
                raise RuntimeError(
                    "invalid SDD GDTS evaluation-consistency receipt")
            summary=json.loads(summary_path.read_text(encoding="utf-8"))
            protocol=json.loads(protocol_path.read_text(encoding="utf-8"))
            if (summary.get("status") != "COMPLETE" or
                    protocol.get("status") != "FROZEN"):
                raise RuntimeError("SDD GDTS baseline protocol is not frozen")
            return

        if receipt.get("output_dir"):
            output=Path(receipt["output_dir"])
        else:
            stamp=datetime.now(ZoneInfo("Asia/Shanghai")).strftime(
                "%Y%m%d_%H%M%S")
            output=(self.root.parents[1]/"analyses"/
                    f"sdd_gdts_eval_consistency_{stamp}")
        atomic_json(receipt_path,{
            "schema":"sdd-gdts-eval-consistency-receipt-v1",
            "status":"RUNNING","output_dir":str(output),
            "checkpoint":str(base),"checkpoint_sha256":digest,
            "manifest":str(self.valid),
            "manifest_sha256":sha256(self.valid),
            "formal_results":str(formal_result),
            "source_commit":self.commit,
        })
        training_log=self.root/"logs/matched_gdts.log"
        try:
            self.run("SDD_GDTS_EVAL_CONSISTENCY",[
                str(PYTHON),"scripts/run_sdd_gdts_eval_consistency.py",
                "--recipe",str(RECIPE),"--runtime-root",str(self.runtime),
                "--cache-root",str(self.cache),"--checkpoint",str(base),
                "--checkpoint-sha256",digest,"--manifest",str(self.valid),
                "--formal-results",str(formal_result),
                "--training-log",str(training_log),
                "--output-dir",str(output)])
            summary_path=output/"SUMMARY.json"
            protocol_path=output/"FROZEN_BASELINE_PROTOCOL.json"
            summary=json.loads(summary_path.read_text(encoding="utf-8"))
            protocol=json.loads(protocol_path.read_text(encoding="utf-8"))
            if (summary.get("status") != "COMPLETE" or
                    protocol.get("status") != "FROZEN"):
                raise RuntimeError(
                    "SDD GDTS consistency audit did not freeze a protocol")
        except BaseException as error:
            atomic_json(receipt_path,{
                "schema":"sdd-gdts-eval-consistency-receipt-v1",
                "status":"FAILED","output_dir":str(output),
                "checkpoint":str(base),"checkpoint_sha256":digest,
                "manifest":str(self.valid),
                "manifest_sha256":sha256(self.valid),
                "formal_results":str(formal_result),
                "source_commit":self.commit,
                "reason":f"{type(error).__name__}: {error}",
            })
            raise
        atomic_json(receipt_path,{
            "schema":"sdd-gdts-eval-consistency-receipt-v1",
            "status":"COMPLETE","output_dir":str(output),
            "checkpoint":str(base),"checkpoint_sha256":digest,
            "manifest":str(self.valid),
            "manifest_sha256":sha256(self.valid),
            "formal_results":str(formal_result),
            "summary":str(summary_path),
            "summary_sha256":sha256(summary_path),
            "frozen_protocol":str(protocol_path),
            "frozen_protocol_sha256":sha256(protocol_path),
            "source_commit":self.commit,
        })

    def config(self,goal,base,base_sha,result):
        target=self.root/"00_identity/RESOLVED_SDD_JMM_UNIFIED.json"; config=json.loads(BASE_CONFIG.read_text()); matched=json.loads(result.read_text()); run=next(x for x in matched["runs"] if x["seed"]==2035); recipe=yaml.safe_load(RECIPE.read_text())
        config["seed"]=3101; config["experiment"].update({"fold":"sdd_trajnet_jmm","source_commit":self.commit,"implementation_base":self.commit[:7]})
        config["paths"].update({"train_manifest":str(self.train),"validation_manifest":str(self.valid),"test_manifest":str(self.train.parents[1]/"test/test_manifest.json"),"accepted_fold_matched_gdts_unet":str(goal),"accepted_fold_matched_gdts_unet_sha256":sha256(goal),"matched_gdts_validation_checkpoint":str(base),"matched_gdts_validation_checkpoint_sha256":base_sha,"matched_gdts_validation_result":str(result),"matched_gdts_validation_result_sha256":sha256(result),"output_root":str(self.root/"06_ebjd_training")})
        config["evaluation"].update({"unit":"pixel","worlds":20,"steps":20,"inference_seeds":list(SEEDS)}); config["evaluation"]["matched_gdts_validation"]={"status":"bound_to_manifest","fold":"sdd_trajnet_jmm","evaluation_seed":2035,"worlds":20,"steps":20,"unit":"pixel","checkpoint_sha256":base_sha,"manifest_sha256":sha256(self.valid),"scene_count":1999,"agent_occurrence_count":2829,**run["metrics"]}
        config["training"].update({"epochs":30,"rollout_training_enabled":False,"paired_marginal_actual_step_projection_enabled":False,"early_stopping":recipe["ebjd_training"]["early_stopping"]}); atomic_json(target,config)
        atomic_json(self.root/"CONFIG_HASHES.json",{"resolved_config":str(target),"resolved_config_sha256":sha256(target),"recipe":str(RECIPE),"recipe_sha256":sha256(RECIPE),"source_commit":self.commit}); return target

    @staticmethod
    def _runtime_recommendation(case):
        correctness=case["correctness"]
        baseline=float(correctness["legacy_repeat_loss"]["max_relative"])
        candidate=float(correctness["candidate_vs_legacy_loss"]["max_relative"])
        numerical_limit=max(1e-3,3.0*baseline+1e-6)
        correct=(bool(correctness["finite"]) and
                 bool(correctness["all_required_gradients_present"]) and
                 candidate <= numerical_limit)
        speedups=[float(value) for value in case["paired_speedups"]]
        speed_ok=bool(speedups) and min(speedups) >= 1.03
        return {
            "runtime": ("compact_uint8_gpu_rigid_v1" if correct and speed_ok
                        else "legacy_cpu_rigid"),
            "correctness_pass":correct,
            "candidate_loss_max_relative":candidate,
            "calibrated_loss_limit":numerical_limit,
            "paired_speedups":speedups,
            "mean_speedup":sum(speedups)/len(speedups),
            "speed_pass":speed_ok,
        }

    def input_pipeline_acceptance(self, config, train_bank):
        INPUT_ACCEPTANCE.mkdir(parents=True,exist_ok=True)
        experiment=Path(
            "/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/"
            "RSJG/RSJG_EBJD_assets/experiments")
        cases=[
            {
                "name":"eth", "correctness_updates":8,
                "measured_updates":16, "repeats":2,
                "config":str(experiment/"eth_unified_coupling_bridge_v1_20261030_104912/configs/RESOLVED_ETH_UNIFIED.json"),
                "checkpoint":str(experiment/"eth_unified_coupling_bridge_v1_20261030_104912/training/best_jade.pt"),
                "train_bank":str(experiment/"eth_unified_coupling_bridge_v1_20261030_104912/proposal_banks/native_train/gdts_proposal_train_manifest_seed3101.npz"),
            },
            {
                "name":"hotel", "correctness_updates":8,
                "measured_updates":32, "repeats":2,
                "config":str(experiment/"hotel_unified_coupling_bridge_v1_20261030_211601/configs/RESOLVED_HOTEL_UNIFIED.json"),
                "checkpoint":str(experiment/"hotel_unified_coupling_bridge_v1_20261030_211601/training/best_jade.pt"),
                "train_bank":str(experiment/"hotel_unified_coupling_bridge_v1_20261030_211601/proposal_banks/native_train/gdts_proposal_train_manifest_seed3101.npz"),
            },
            {
                "name":"univ", "correctness_updates":8,
                "measured_updates":32, "repeats":2,
                "config":str(experiment/"univ_unified_coupling_bridge_v1_20261031_111541/configs/RESOLVED_UNIV_UNIFIED.json"),
                "checkpoint":str(experiment/"univ_unified_coupling_bridge_v1_20261031_111541/training/best_jade.pt"),
                "train_bank":str(experiment/"univ_unified_coupling_bridge_v1_20261031_111541/proposal_banks/native_train/gdts_proposal_train_manifest_seed3101.npz"),
            },
            {
                "name":"zara1", "correctness_updates":8,
                "measured_updates":16, "repeats":2,
                "config":str(experiment/"zara1_unified_coupling_bridge_v1_20261028_214026/configs/model_config.json"),
                "checkpoint":str(experiment/"zara1_unified_coupling_bridge_v1_20261028_214026/training/best_jade.pt"),
                "train_bank":str(experiment/"zara1_unified_coupling_bridge_v1_20261028_214026/proposal_banks/train_seed3101.npz"),
            },
            {
                "name":"zara2", "correctness_updates":8,
                "measured_updates":16, "repeats":2,
                "config":str(experiment/"zara2_unified_coupling_bridge_v1_20261029_094715/configs/RESOLVED_ZARA2_UNIFIED.json"),
                "checkpoint":str(experiment/"zara2_unified_coupling_bridge_v1_20261029_094715/training/best_jade.pt"),
                "train_bank":str(experiment/"zara2_unified_coupling_bridge_v1_20261029_094715/proposal_banks/native_train/gdts_proposal_train_seed3101.npz"),
            },
            {
                "name":"sdd_jmm", "correctness_updates":8,
                "measured_updates":16, "repeats":2,
                "config":str(config),
                # Benchmark-only same-architecture state.  It never becomes a
                # formal SDD initialization or selector candidate.
                "checkpoint":str(experiment/"zara1_unified_coupling_bridge_v1_20261028_214026/training/best_jade.pt"),
                "train_bank":str(train_bank),
                "checkpoint_role":"benchmark_only_same_architecture_state",
            },
        ]
        for case in cases:
            for key in ("config","checkpoint","train_bank"):
                if not Path(case[key]).exists():
                    raise RuntimeError(
                        f"input acceptance asset absent: {case['name']} {key}")
        spec=INPUT_ACCEPTANCE/"BENCHMARK_SPEC.json"
        atomic_json(spec,{"schema":"ebjd-input-pipeline-benchmark-spec-v2",
                          "source_commit":self.commit,"cases":cases})
        results=INPUT_ACCEPTANCE/"BENCHMARK_RESULTS.json"
        fallback_reason=None
        if not results.exists():
            try:
                self.run("EBJD_INPUT_PIPELINE_ACCEPTANCE",[
                    str(PYTHON),"-u","scripts/benchmark_ebjd_input_pipeline.py",
                    "--spec",str(spec),"--output",str(INPUT_ACCEPTANCE),
                    "--correctness-updates","8","--warmup-updates","4",
                    "--measured-updates","16","--repeats","2"])
            except BaseException as error:
                fallback_reason=f"{type(error).__name__}: {error}"
        recommendations={}
        if results.exists():
            payload=json.loads(results.read_text(encoding="utf-8"))
            if payload.get("status") != "COMPLETE":
                fallback_reason="benchmark receipt is not COMPLETE"
            else:
                recommendations={case["name"]:self._runtime_recommendation(case)
                                 for case in payload["cases"]}
        if not recommendations:
            recommendations={case["name"]:{
                "runtime":"legacy_cpu_rigid","correctness_pass":False,
                "speed_pass":False,"fallback_reason":fallback_reason,
            } for case in cases}
        selected=recommendations["sdd_jmm"]["runtime"]
        resolved=json.loads(Path(config).read_text(encoding="utf-8"))
        resolved["training"]["ebjd_input_pipeline"]=selected
        resolved["training"]["ebjd_input_pipeline_receipt"]=str(
            self.root/"receipts/EBJD_INPUT_PIPELINE_ACCEPTANCE.json")
        atomic_json(config,resolved)
        receipt={
            "schema":"ebjd-input-pipeline-acceptance-v1",
            "source_commit":self.commit,"benchmark_results":str(results),
            "recommendations":recommendations,
            "sdd_selected_runtime":selected,
            "fallback_reason":fallback_reason,
            "formal_checkpoint_or_optimizer_modified":False,
        }
        atomic_json(
            self.root/"receipts/EBJD_INPUT_PIPELINE_ACCEPTANCE.json",receipt)
        atomic_json(self.root/"CONFIG_HASHES.json",{
            "resolved_config":str(config),
            "resolved_config_sha256":sha256(config),
            "recipe":str(RECIPE),"recipe_sha256":sha256(RECIPE),
            "source_commit":self.commit,
            "ebjd_input_pipeline":selected,
            "acceptance_receipt_sha256":sha256(
                self.root/"receipts/EBJD_INPUT_PIPELINE_ACCEPTANCE.json")})
        if results.exists():
            report_dir=INPUT_ACCEPTANCE/"reports/fivefold_acceptance"
            subprocess.run([
                str(PYTHON),"scripts/report_ebjd_input_pipeline.py",
                "--results",str(results),"--audit",
                str(INPUT_ACCEPTANCE/"DATA_PATH_AUDIT.json"),
                "--output-dir",str(report_dir)],cwd=PAPER,check=True)
        self.state(
            "EBJD_INPUT_PIPELINE_ACCEPTANCE","COMPLETE",
            selected_runtime=selected,receipt=str(
                self.root/"receipts/EBJD_INPUT_PIPELINE_ACCEPTANCE.json"))
        return config

    def execute(self):
        self.state("VERIFY_DATA","RUNNING"); self.verify(); self.wait_gpu(); self.caches(); goal,base=self.gdts(); bank,pattern,digest,result=self.assets(base); config=self.config(goal,base,digest,result); config=self.input_pipeline_acceptance(config,bank)
        if not (self.root/"06_ebjd_training/FINAL_EVALUATION.json").exists(): self.run("EBJD_UNIFIED_TRAIN_EVAL",[str(PYTHON),"-u","scripts/run_zara1_unified_coupling_bridge.py","--config",str(config),"--fold","sdd_trajnet_jmm","--compact-scenes","--train-bank",str(bank),"--validation-bank-pattern",pattern,"--output",str(self.root/"06_ebjd_training"),"--source-commit",self.commit,"--training-seed","3101","--provider-seed","3101","--max-gpu-seconds","604800","--final-reserve-seconds","14400","--resume"])
        checkpoints={str(p):sha256(p) for p in (goal,base,self.root/"06_ebjd_training/best_jade.pt",self.root/"06_ebjd_training/last.pt") if p.exists()}; atomic_json(self.root/"CHECKPOINT_MANIFEST.json",checkpoints); self.state("COMPLETE","COMPLETE")

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--root",required=True); parser.add_argument("--source-commit",required=True); parser.add_argument("--migration-reason"); parser.add_argument("--migration-optimizer-updates",type=int); args=parser.parse_args()
    root=Path(args.root).resolve(); root.mkdir(parents=True,exist_ok=True)
    lock_path=root/"SUPERVISOR.lock"
    with lock_path.open("a+",encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another SDD supervisor owns the pipeline lock") from error
        lock.seek(0); lock.truncate(); lock.write(f"{os.getpid()}\n"); lock.flush(); os.fsync(lock.fileno())
        Pipeline(
            root, args.source_commit,
            migration_reason=args.migration_reason,
            migration_optimizer_updates=args.migration_optimizer_updates,
        ).execute()
if __name__=="__main__": main()
