#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "$project_root/.." && pwd)"
workspace_root="$(cd "$repo_root/.." && pwd)"
python_bin="$workspace_root/.conda/rsjg/bin/python"
source_assets="$workspace_root/RSJG_EBJD_diagnostic_assets/univ_no_token_transfer_20261021"
assets="$workspace_root/RSJG_EBJD_diagnostic_assets/univ_epoch81_frozen_eval_20261021"
archive="$project_root/docs/validation/ebjd_univ_epoch81_frozen_eval_20261021"
protocol="$project_root/docs/validation/ebjd_univ_no_token_transfer_20261021"
control_config="$protocol/CONTROL_CONFIG.json"
candidate_config="$protocol/CANDIDATE_CONFIG.json"
control_checkpoint="$source_assets/runs/univ/3101/rollout_mechanism_off/univ_transfer_control_e74_81_20261021/epoch_081.pt"
candidate_checkpoint="$source_assets/runs/univ/3101/v2_no_token/univ_transfer_no_token_e74_81_20261021/epoch_081.pt"
control_receipts="$assets/receipts/control"
candidate_receipts="$assets/receipts/candidate"
working="$assets/working"
ledger="$assets/GPU_LEDGER.jsonl"
budget=7200

mkdir -p "$control_receipts" "$candidate_receipts" "$working"
touch "$ledger"
cd "$project_root"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -n "$(git -C "$repo_root" status --porcelain -- ebjd_clean)" ]]; then
  echo "formal frozen evaluation refuses a dirty ebjd_clean tree" >&2
  exit 20
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[0-9]+'; then
  echo "another GPU compute process is active" >&2
  exit 21
fi

spent_seconds() {
  "$python_bin" - "$ledger" <<'PY'
import json,sys
print(sum(float(json.loads(x)["elapsed_seconds"])
          for x in open(sys.argv[1]) if x.strip()))
PY
}

estimate_seconds() {
  "$python_bin" - "$ledger" "$1" <<'PY'
import json,sys
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
values=[float(x["elapsed_seconds"]) for x in rows
        if x["arm"] == sys.argv[2] and x["status"] == "COMPLETE"]
default=720.0 if sys.argv[2] == "candidate" else 420.0
print(sum(values)/len(values) if values else default)
PY
}

record_invocation() {
  "$python_bin" - "$ledger" "$1" "$2" "$3" "$4" <<'PY'
import json,os,sys,tempfile
from pathlib import Path
path=Path(sys.argv[1]); rows=path.read_text().splitlines()
rows.append(json.dumps({"arm":sys.argv[2], "seed":int(sys.argv[3]),
                        "elapsed_seconds":float(sys.argv[4]),
                        "status":sys.argv[5]}, sort_keys=True))
fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
with os.fdopen(fd,'w') as f: f.write('\n'.join(rows)+'\n')
os.replace(tmp,path)
PY
}

can_start() {
  local arm="$1" spent estimate remaining required
  spent="$(spent_seconds)"
  estimate="$(estimate_seconds "$arm")"
  "$python_bin" - "$spent" "$estimate" "$budget" <<'PY'
import sys
spent,estimate,budget=map(float,sys.argv[1:])
# Retain a 15% runtime margin plus 30 seconds of process teardown room.
raise SystemExit(0 if budget-spent >= estimate*1.15+30.0 else 1)
PY
}

run_seed() {
  local arm="$1" seed="$2" config checkpoint ablation receipts raw fixed time_file log_file
  if [[ "$arm" == candidate ]]; then
    config="$candidate_config"; checkpoint="$candidate_checkpoint"
    ablation=v2_no_token; receipts="$candidate_receipts"
  else
    config="$control_config"; checkpoint="$control_checkpoint"
    ablation=rollout_mechanism_off; receipts="$control_receipts"
  fi
  if [[ -f "$receipts/seed_${seed}.json" ]] && \
      "$python_bin" - "$receipts/seed_${seed}.json" "$config" "$checkpoint" \
        "$project_root/ebjd/evaluate.py" "$seed" "$arm" <<'PY'
import hashlib,json,sys
from pathlib import Path
from ebjd.evaluate import valid_seed_receipt
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
receipt=json.load(open(sys.argv[1])); config=json.load(open(sys.argv[2]))
manifest=config['paths']['validation_manifest']
expected={'checkpoint_sha256':sha(sys.argv[3]),'config_sha256':sha(sys.argv[2]),
          'manifest_sha256':sha(manifest),'evaluator_sha256':sha(sys.argv[4]),
          'seed':int(sys.argv[5]),'worlds':20,'steps':20,'ddim_eta':0.0,
          'scene_count':947,'agent_occurrence_count':24334}
raise SystemExit(0 if valid_seed_receipt(
    receipt,expected,require_fixed=sys.argv[6]=='candidate') else 1)
PY
  then
    echo "reused validated receipt: $arm seed $seed"
    return 0
  fi
  raw="$working/${arm}_seed${seed}_raw.json"
  fixed="$working/${arm}_seed${seed}_fixed.json"
  time_file="$working/${arm}_seed${seed}.seconds"
  log_file="$working/${arm}_seed${seed}.log"
  args=("$python_bin" -u -B -m ebjd.evaluate
        --config "$config" --checkpoint "$checkpoint"
        --ablation "$ablation" --split validation --device cuda:0
        --inference-seeds "$seed" --seed-receipt-dir "$receipts"
        --output "$raw")
  if [[ "$arm" == candidate ]]; then
    args+=(--fixed-anchor-output "$fixed")
  fi
  set +e
  /usr/bin/time -f '%e' -o "$time_file" "${args[@]}" >"$log_file" 2>&1
  status=$?
  set -e
  elapsed="$(tail -n 1 "$time_file")"
  if [[ "$status" -eq 0 && -f "$receipts/seed_${seed}.json" ]]; then
    label=COMPLETE
  else
    label=FAILED
  fi
  record_invocation "$arm" "$seed" "$elapsed" "$label"
  printf '%s seed=%s status=%s elapsed=%s cumulative=%s\n' \
    "$arm" "$seed" "$label" "$elapsed" "$(spent_seconds)"
  if [[ "$label" != COMPLETE ]]; then
    return 1
  fi
}

order=("candidate:2035" "control:2035"
       "candidate:2036" "candidate:2037" "candidate:2038" "candidate:2039"
       "control:2036" "control:2037" "control:2038" "control:2039")
for item in "${order[@]}"; do
  arm="${item%%:*}"; seed="${item##*:}"
  if ! can_start "$arm"; then
    printf 'PARTIAL_BUDGET_GUARD arm=%s seed=%s spent=%s budget=%s\n' \
      "$arm" "$seed" "$(spent_seconds)" "$budget" | tee "$assets/PIPELINE_STATE.txt"
    exit 22
  fi
  run_seed "$arm" "$seed"
done

seeds=(2035 2036 2037 2038 2039)
control_result="$assets/CONTROL_EPOCH081_RAW_SEEDS2035_2039.json"
candidate_raw="$assets/CANDIDATE_EPOCH081_RAW_SEEDS2035_2039.json"
candidate_fixed="$assets/CANDIDATE_EPOCH081_FIXED_SEEDS2035_2039.json"
"$python_bin" -u -B -m ebjd.evaluate \
  --config "$control_config" --checkpoint "$control_checkpoint" \
  --ablation rollout_mechanism_off --split validation --device cpu \
  --inference-seeds "${seeds[@]}" --seed-receipt-dir "$control_receipts" \
  --output "$control_result"
"$python_bin" -u -B -m ebjd.evaluate \
  --config "$candidate_config" --checkpoint "$candidate_checkpoint" \
  --ablation v2_no_token --split validation --device cpu \
  --inference-seeds "${seeds[@]}" --seed-receipt-dir "$candidate_receipts" \
  --output "$candidate_raw" --fixed-anchor-output "$candidate_fixed"

"$python_bin" -u -B scripts/finalize_univ_epoch81_frozen_eval.py \
  --asset-root "$assets" --archive "$archive" \
  --control-result "$control_result" \
  --candidate-raw-result "$candidate_raw" \
  --candidate-fixed-result "$candidate_fixed" \
  --parent-result "$project_root/docs/experiments/2026-10-13_univ_main_seed3101/EBJD_UNIV_SEEDS2035_2039.json" \
  --gdts-comparison "$project_root/docs/experiments/2026-10-13_univ_main_seed3101/UNIV_FINAL_COMPARISON.json" \
  --control-config "$control_config" --candidate-config "$candidate_config"

git -C "$repo_root" add ebjd_clean/docs/validation/ebjd_univ_epoch81_frozen_eval_20261021
git -C "$repo_root" commit -m 'Archive UNIV epoch81 frozen five-seed evaluation'
git -C "$repo_root" push origin HEAD:research/ebjd-univ-epoch81-frozen-eval-v1
local_sha="$(git -C "$repo_root" rev-parse HEAD)"
remote_sha="$(git -C "$repo_root" ls-remote origin refs/heads/research/ebjd-univ-epoch81-frozen-eval-v1 | cut -f1)"
printf 'COMPLETE\nlocal_sha=%s\nremote_sha=%s\ngpu_seconds=%s\n' \
  "$local_sha" "$remote_sha" "$(spent_seconds)" | tee "$assets/PIPELINE_STATE.txt"
test "$local_sha" = "$remote_sha"
