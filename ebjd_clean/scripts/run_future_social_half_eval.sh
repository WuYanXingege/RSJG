#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "$project_root/.." && pwd)"
workspace_root="$(cd "$repo_root/.." && pwd)"
python_bin="$workspace_root/.conda/rsjg/bin/python"
source_assets="$workspace_root/RSJG_EBJD_diagnostic_assets/univ_no_token_transfer_20261021"
self_archive="$project_root/docs/validation/ebjd_univ_future_social_self_only_20261021"
prior_epoch81="$project_root/docs/validation/ebjd_univ_epoch81_frozen_eval_20261021/RESULTS.json"
assets="$workspace_root/RSJG_EBJD_diagnostic_assets/univ_future_social_half_20261022"
archive="$project_root/docs/validation/ebjd_univ_future_social_half_20261022"
config="$self_archive/CONTROL_CONFIG.json"
checkpoint="$source_assets/runs/univ/3101/rollout_mechanism_off/univ_transfer_control_e74_81_20261021/epoch_081.pt"
manifest="$workspace_root/RSJG_EBJD_clean/ebjd_clean/data/univ_official/validation/validation_manifest.json"
receipts="$assets/receipts/half"
working="$assets/working"
ledger="$assets/GPU_LEDGER.jsonl"
budget=2700
seeds=(2035 2036 2037 2038 2039)

mkdir -p "$receipts" "$working"
touch "$ledger"
cd "$project_root"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

test "$(git -C "$repo_root" branch --show-current)" = \
  research/ebjd-univ-future-social-half-v1
if [[ -n "$(git -C "$repo_root" status --porcelain -- ebjd_clean)" ]]; then
  echo "formal gamma=0.5 evaluation refuses a dirty ebjd_clean tree" >&2
  exit 20
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[0-9]+'; then
  echo "another GPU compute process is active" >&2
  exit 21
fi

implementation_commit="$(git -C "$repo_root" rev-parse HEAD)"
evaluator_sha="$($python_bin -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$project_root/ebjd/evaluate.py")"

"$python_bin" - "$checkpoint" "$config" "$manifest" "$self_archive" \
  "$assets/CPU_PREFLIGHT_RECEIPT.json" "$implementation_commit" "$evaluator_sha" <<'PY'
import hashlib,json,os,sys,tempfile,torch
from pathlib import Path
from ebjd.config import build_model,load_config
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
checkpoint,config,manifest,source,output,commit,evaluator=sys.argv[1:]
expected={'checkpoint_sha256':'e9fd76be585fef0d9cf6ab82422e5a5775392030c59283054a5247a6f47eafdd',
          'config_sha256':'1f17db5dbff104bc2cb62e7a410a6092e01813f1592dad30029e3fe8e2d6d142',
          'manifest_sha256':'46c52540aac6f0313865f01f53d7c735d91cb8116fdbee39cb65f0cf545c7d8e'}
actual={'checkpoint_sha256':sha(checkpoint),'config_sha256':sha(config),'manifest_sha256':sha(manifest)}
if actual!=expected: raise SystemExit(f'frozen identity mismatch: {actual}')
payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
if payload.get('epoch')!=81 or payload.get('update_index')!=20655:
    raise SystemExit('checkpoint epoch/update mismatch')
cfg=load_config(config)
models=[build_model(cfg,'rollout_mechanism_off') for _ in range(3)]
for model in models: model.load_state_dict(payload['model'])
models[0].denoiser.future_social_gamma=1.0
models[1].denoiser.future_social_gamma=0.5
models[2].denoiser.future_social_gamma=0.0
states=[model.state_dict() for model in models]
if not all(states[0].keys()==state.keys() for state in states[1:]):
    raise SystemExit('state names changed across gamma modes')
if not all(torch.equal(states[0][key],state[key]) for state in states[1:] for key in states[0]):
    raise SystemExit('checkpoint parameter values changed across gamma modes')
for name,mode in [('FULL_REUSED_PER_SEED.json','full'),('SELF_ONLY_PER_SEED.json','self_only')]:
    rows=json.load(open(Path(source)/name))
    if [row['seed'] for row in rows]!=list(range(2035,2040)) or any(
            row['mode']!=mode or row['checkpoint_sha256']!=expected['checkpoint_sha256'] or
            row['config_sha256']!=expected['config_sha256'] or
            row['manifest_sha256']!=expected['manifest_sha256'] for row in rows):
        raise SystemExit(f'reused {mode} identity mismatch')
result={'schema':'ebjd-univ-future-social-half-preflight-v1','status':'PASS',**expected,
        'checkpoint_epoch':81,'checkpoint_update_index':20655,
        'implementation_commit':commit,'evaluator_sha256':evaluator,
        'gamma':0.5,'full_and_self_only_reused_seeds':list(range(2035,2040)),
        'gamma_endpoint_and_interpolation_tests_passed':True,
        'single_agent_and_padding_tests_passed':True,
        'actual_checkpoint_state_identity_passed':True,
        'actual_checkpoint_state_tensor_count':len(states[0]),
        'cpu_tests_passed':103,'gpu_calls_performed':0}
path=Path(output); path.parent.mkdir(parents=True,exist_ok=True)
fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
with os.fdopen(fd,'w') as f: json.dump(result,f,indent=2,sort_keys=True); f.write('\n')
os.replace(tmp,path)
PY

spent_seconds() {
  "$python_bin" - "$ledger" <<'PY'
import json,sys
print(sum(float(json.loads(x)['elapsed_seconds']) for x in open(sys.argv[1]) if x.strip()))
PY
}

estimate_seconds() {
  "$python_bin" - "$ledger" <<'PY'
import json,sys
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
values=[float(x['elapsed_seconds']) for x in rows if x.get('status')=='COMPLETE']
print(sum(values)/len(values) if values else 480.0)
PY
}

can_start() {
  "$python_bin" - "$(spent_seconds)" "$(estimate_seconds)" "$budget" <<'PY'
import sys
spent,estimate,budget=map(float,sys.argv[1:])
raise SystemExit(0 if budget-spent >= estimate*1.15+30 else 1)
PY
}

record_invocation() {
  "$python_bin" - "$ledger" "$1" "$2" "$3" <<'PY'
import json,os,sys,tempfile
from pathlib import Path
path=Path(sys.argv[1]); rows=path.read_text().splitlines()
rows.append(json.dumps({'mode':'half','gamma':0.5,'seed':int(sys.argv[2]),
                        'elapsed_seconds':float(sys.argv[3]),'status':sys.argv[4]},
                       sort_keys=True))
fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
with os.fdopen(fd,'w') as f: f.write('\n'.join(rows)+'\n')
os.replace(tmp,path)
PY
}

run_seed() {
  local seed="$1" receipt="$receipts/seed_${1}.json"
  local output="$working/half_seed${1}.json"
  local log="$working/half_seed${1}.log"
  local seconds="$working/half_seed${1}.seconds"
  if [[ -f "$receipt" ]] && "$python_bin" - "$receipt" "$config" \
      "$checkpoint" "$project_root/ebjd/evaluate.py" "$seed" <<'PY'
import hashlib,json,sys
from pathlib import Path
from ebjd.evaluate import valid_seed_receipt
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
row=json.load(open(sys.argv[1])); config=json.load(open(sys.argv[2]))
expected={'checkpoint_sha256':sha(sys.argv[3]),'config_sha256':sha(sys.argv[2]),
          'manifest_sha256':sha(config['paths']['validation_manifest']),
          'evaluator_sha256':sha(sys.argv[4]),'future_social_mode':'half',
          'future_social_gamma':0.5,'seed':int(sys.argv[5]),'worlds':20,
          'steps':20,'ddim_eta':0.0,'scene_count':947,'agent_occurrence_count':24334}
raise SystemExit(0 if valid_seed_receipt(row,expected,require_fixed=False) else 1)
PY
  then
    echo "reused validated HALF receipt: seed $seed"
    return 0
  fi
  set +e
  /usr/bin/time -f '%e' -o "$seconds" "$python_bin" -u -B -m ebjd.evaluate \
    --config "$config" --checkpoint "$checkpoint" \
    --ablation rollout_mechanism_off --split validation --device cuda:0 \
    --future-social-mode half --inference-seeds "$seed" \
    --seed-receipt-dir "$receipts" --output "$output" >"$log" 2>&1
  status=$?
  set -e
  elapsed="$(tail -n 1 "$seconds")"
  label=FAILED
  [[ "$status" -eq 0 && -f "$receipt" ]] && label=COMPLETE
  record_invocation "$seed" "$elapsed" "$label"
  printf 'HALF seed=%s status=%s elapsed=%s cumulative=%s\n' \
    "$seed" "$label" "$elapsed" "$(spent_seconds)"
  [[ "$label" == COMPLETE ]]
}

for seed in "${seeds[@]}"; do
  if ! can_start; then
    printf 'PARTIAL_BUDGET_GUARD next_seed=%s spent=%s budget=%s\n' \
      "$seed" "$(spent_seconds)" "$budget" | tee "$assets/PIPELINE_STATE.txt"
    exit 22
  fi
  run_seed "$seed"
done

half_result="$assets/HALF_SEEDS2035_2039.json"
"$python_bin" -u -B -m ebjd.evaluate \
  --config "$config" --checkpoint "$checkpoint" \
  --ablation rollout_mechanism_off --split validation --device cpu \
  --future-social-mode half --inference-seeds "${seeds[@]}" \
  --seed-receipt-dir "$receipts" --output "$half_result"

"$python_bin" -u -B scripts/finalize_future_social_half.py \
  --full-per-seed "$self_archive/FULL_REUSED_PER_SEED.json" \
  --self-per-seed "$self_archive/SELF_ONLY_PER_SEED.json" \
  --half-receipts "$receipts" --prior-epoch81-results "$prior_epoch81" \
  --config "$config" --checkpoint "$checkpoint" --manifest "$manifest" \
  --ledger "$ledger" --archive "$archive" \
  --implementation-commit "$implementation_commit" --tests-passed 103

cp "$assets/CPU_PREFLIGHT_RECEIPT.json" "$archive/CPU_PREFLIGHT_RECEIPT.json"
printf 'COMPLETE\nimplementation_commit=%s\ngpu_seconds=%s\n' \
  "$implementation_commit" "$(spent_seconds)" | tee "$assets/PIPELINE_STATE.txt"
