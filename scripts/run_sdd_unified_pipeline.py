#!/usr/bin/env python3
"""Resumable JMM TrajNet-SDD asset preparation and unified EBJD pipeline."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, tempfile, time
from datetime import datetime, timezone
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
PYTHON = Path("/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/.conda/rsjg/bin/python")
RECIPE = PAPER / "configs/asset_preparation/sdd_jmm_gdts.yaml"
BASE_CONFIG = PAPER / "configs/resolved/zara1.json"
SEEDS = tuple(range(2035, 2040))

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
    def __init__(self, root, commit):
        self.root, self.commit = Path(root).resolve(), commit
        self.runtime = self.root / "04_matched_gdts/runtime"
        self.cache = self.root / "04_matched_gdts/cache"
        data = Path("/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_assets/data/sdd_trajnet_jmm/manifests")
        self.train, self.valid = data / "train/train_manifest.json", data / "validation/validation_manifest.json"
        for name in ("00_identity","01_map_recovery","02_data_manifests","03_goal_pretrain","04_matched_gdts","05_proposal_banks","06_ebjd_training","07_evaluation","logs","reports","receipts"):
            (self.root/name).mkdir(parents=True, exist_ok=True)
        self.state_path, self.budget_path = self.root/"PIPELINE_STATE.json", self.root/"GPU_BUDGET.json"
        if not self.budget_path.exists(): atomic_json(self.budget_path, {"limit_seconds":604800.0,"consumed_seconds":0.0,"stages":[]})

    def state(self, stage, status, **extra):
        atomic_json(self.state_path, {"schema":"ebjd-sdd-jmm-unified-pipeline-v1","stage":stage,"status":status,"source_commit":self.commit,"experiment_root":str(self.root),"updated_at_utc":datetime.now(timezone.utc).isoformat(),**extra})

    def run(self, stage, command):
        log = self.root/"logs"/(stage.lower()+".log"); self.state(stage,"RUNNING",command=command,log=str(log))
        started=time.perf_counter(); outcome="failed"
        try:
            with log.open("a",encoding="utf-8") as handle:
                result=subprocess.run(command,cwd=PAPER,stdout=handle,stderr=subprocess.STDOUT,check=False)
            if result.returncode: raise RuntimeError(f"{stage} exited {result.returncode}")
            outcome="complete"
        finally:
            budget=json.loads(self.budget_path.read_text()); seconds=time.perf_counter()-started
            budget["consumed_seconds"]+=seconds; budget["stages"].append({"stage":stage,"seconds":seconds,"status":outcome}); atomic_json(self.budget_path,budget)

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
        common=[str(PYTHON),"scripts/run_legacy_gdts_stage.py","--recipe",str(RECIPE),"--runtime-root",str(self.runtime),"--batch-cache-root",str(self.cache)]
        if not (self.cache/"sdd/sdd/data_batches/finished_train_batches.txt").exists(): self.run("CACHE_INDEPENDENT",common+["--stage","scene_cache_independent"])
        if not (self.cache/"sdd/sdd/data_batches_joint_v2/finished_train_batches.txt").exists(): self.run("CACHE_JOINT",common+["--stage","scene_cache"])
        for split,manifest in (("train",self.train),("valid",self.valid)):
            receipt=self.root/f"receipts/CACHE_PARITY_{split.upper()}.json"
            if not receipt.exists(): self.run("AUDIT_CACHE_"+split.upper(),[str(PYTHON),"scripts/audit_manifest_gdts_cache.py","--recipe",str(RECIPE),"--manifest",str(manifest),"--runtime-root",str(self.runtime),"--cache-dir",str(self.cache/f"sdd/sdd/data_batches_joint_v2/{split}_batches"),"--output",str(receipt)])

    def gdts(self):
        common=[str(PYTHON),"scripts/run_legacy_gdts_stage.py","--recipe",str(RECIPE),"--runtime-root",str(self.runtime),"--batch-cache-root",str(self.cache)]
        goal=self.runtime/"output/sdd/runs/sdd_jmm_goal_pretrain_seed3101_v1/saved_models/goal_pretrain_best_model.pt"
        if not goal.exists(): self.run("GOAL_PRETRAIN",common+["--stage","goal"])
        if not goal.exists(): raise RuntimeError("goal selected checkpoint absent")
        base=self.runtime/"output/sdd/runs/sdd_jmm_matched_gdts_seed3101_v1/saved_models/best_model.pt"
        if not base.exists(): self.run("MATCHED_GDTS",common+["--stage","baseline","--goal-checkpoint",str(goal)])
        if not base.exists(): raise RuntimeError("matched GDTS selected checkpoint absent")
        return goal,base

    def assets(self, base):
        digest=sha256(base); result=self.root/"04_matched_gdts/evaluation/results.json"
        if not result.exists(): self.run("MATCHED_GDTS_EVAL",[str(PYTHON),"scripts/evaluate_matched_gdts.py","--recipe",str(RECIPE),"--runtime-root",str(self.runtime),"--scene-cache-root",str(self.cache),"--checkpoint",str(base),"--checkpoint-sha256",digest,"--manifest",str(self.valid),"--parity-receipt",str(self.root/"receipts/CACHE_PARITY_VALID.json"),"--output-dir",str(result.parent),"--split","valid","--seeds",*map(str,SEEDS)])
        train_dir=self.root/"05_proposal_banks/train"; train_bank=train_dir/"gdts_proposal_train_seed3101.npz"
        if not train_bank.exists(): self.run("EXPORT_TRAIN_BANK",[str(PYTHON),"scripts/export_zara1_train_proposals.py","--recipe",str(RECIPE),"--checkpoint",str(base),"--checkpoint-sha256",digest,"--manifest",str(self.train),"--manifest-sha256",sha256(self.train),"--scene-cache-root",str(self.cache),"--runtime-root",str(self.runtime),"--output-dir",str(train_dir),"--seed","3101"])
        pattern=self.root/"05_proposal_banks/validation/seed_{seed}/gdts_proposal_B_seed{seed}.npz"
        for seed in SEEDS:
            target=Path(str(pattern).format(seed=seed))
            if not target.exists(): self.run(f"EXPORT_VALIDATION_BANK_{seed}",[str(PYTHON),"scripts/export_zara1_endpoint_banks.py","gdts","--recipe",str(RECIPE),"--runtime-root",str(self.runtime),"--scene-cache-root",str(self.cache),"--checkpoint",str(base),"--checkpoint-sha256",digest,"--manifest",str(self.valid),"--manifest-sha256",sha256(self.valid),"--output-dir",str(target.parent),"--seed",str(seed)])
        return train_bank,str(pattern),digest,result

    def config(self,goal,base,base_sha,result):
        target=self.root/"00_identity/RESOLVED_SDD_JMM_UNIFIED.json"; config=json.loads(BASE_CONFIG.read_text()); matched=json.loads(result.read_text()); run=next(x for x in matched["runs"] if x["seed"]==2035)
        config["seed"]=3101; config["experiment"].update({"fold":"sdd_trajnet_jmm","source_commit":self.commit,"implementation_base":self.commit[:7]})
        config["paths"].update({"train_manifest":str(self.train),"validation_manifest":str(self.valid),"test_manifest":str(self.train.parents[1]/"test/test_manifest.json"),"accepted_fold_matched_gdts_unet":str(goal),"accepted_fold_matched_gdts_unet_sha256":sha256(goal),"matched_gdts_validation_checkpoint":str(base),"matched_gdts_validation_checkpoint_sha256":base_sha,"matched_gdts_validation_result":str(result),"matched_gdts_validation_result_sha256":sha256(result),"output_root":str(self.root/"06_ebjd_training")})
        config["evaluation"].update({"unit":"pixel","worlds":20,"steps":20,"inference_seeds":list(SEEDS)}); config["evaluation"]["matched_gdts_validation"]={"status":"bound_to_manifest","fold":"sdd_trajnet_jmm","evaluation_seed":2035,"worlds":20,"steps":20,"unit":"pixel","checkpoint_sha256":base_sha,"manifest_sha256":sha256(self.valid),"scene_count":1999,"agent_occurrence_count":2829,**run["metrics"]}
        config["training"].update({"epochs":30,"rollout_training_enabled":False,"paired_marginal_actual_step_projection_enabled":False}); atomic_json(target,config)
        atomic_json(self.root/"CONFIG_HASHES.json",{"resolved_config":str(target),"resolved_config_sha256":sha256(target),"recipe":str(RECIPE),"recipe_sha256":sha256(RECIPE),"source_commit":self.commit}); return target

    def execute(self):
        self.state("VERIFY_DATA","RUNNING"); self.verify(); self.wait_gpu(); self.caches(); goal,base=self.gdts(); bank,pattern,digest,result=self.assets(base); config=self.config(goal,base,digest,result)
        if not (self.root/"06_ebjd_training/FINAL_EVALUATION.json").exists(): self.run("EBJD_UNIFIED_TRAIN_EVAL",[str(PYTHON),"-u","scripts/run_zara1_unified_coupling_bridge.py","--config",str(config),"--fold","sdd_trajnet_jmm","--compact-scenes","--train-bank",str(bank),"--validation-bank-pattern",pattern,"--output",str(self.root/"06_ebjd_training"),"--source-commit",self.commit,"--training-seed","3101","--provider-seed","3101","--max-gpu-seconds","604800","--final-reserve-seconds","14400","--resume"])
        checkpoints={str(p):sha256(p) for p in (goal,base,self.root/"06_ebjd_training/best_jade.pt",self.root/"06_ebjd_training/last.pt") if p.exists()}; atomic_json(self.root/"CHECKPOINT_MANIFEST.json",checkpoints); self.state("COMPLETE","COMPLETE")

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--root",required=True); parser.add_argument("--source-commit",required=True); args=parser.parse_args(); Pipeline(args.root,args.source_commit).execute()
if __name__=="__main__": main()
