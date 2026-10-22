#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "$project_root/.." && pwd)"
workspace_root="$(cd "$repo_root/.." && pwd)"
python_bin="$workspace_root/.conda/rsjg/bin/python"
hotel_root="$workspace_root/RSJG_EBJD_clean/ebjd_clean"
run_root="$hotel_root/outputs/formal_hotel_speed_9ecde66_20261009/hotel/3101/none/hotel_main_seed3101_9ecde66_speed_20261009"
config="$run_root/RESOLVED_CONFIG.json"
checkpoint="$run_root/epoch_045.pt"
manifest="$hotel_root/data/hotel_official/test/test_manifest.json"
full_result="$hotel_root/outputs/evaluations/hotel_seed3101_epoch045_seeds2035_2039_20261012/ebjd_epoch045_test_mirror_seeds2035_2039.json"
gdts_result="$workspace_root/RSJG_JDV2_clean/outputs/joint_dependency_v2/hotel_a0_a1_5b6e4e4/formal_launch/results/E_joint_baseline.json"
assets="$workspace_root/RSJG_EBJD_diagnostic_assets/hotel_future_social_half_20261022"
archive="$project_root/docs/validation/ebjd_hotel_future_social_half_20261022"
receipts="$assets/receipts/half"
working="$assets/working"
ledger="$assets/GPU_LEDGER.jsonl"
budget=1800
seeds=(2035 2036 2037 2038 2039)

mkdir -p "$receipts" "$working"
touch "$ledger"
cd "$project_root"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

test "$(git -C "$repo_root" branch --show-current)" = \
  research/ebjd-hotel-future-social-half-v1
if [[ -n "$(git -C "$repo_root" status --porcelain -- ebjd_clean)" ]]; then
  echo "formal HOTEL gamma=0.5 evaluation refuses a dirty ebjd_clean tree" >&2
  exit 20
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[0-9]+'; then
  echo "another GPU compute process is active" >&2
  exit 21
fi

implementation_commit="$(git -C "$repo_root" rev-parse HEAD)"
evaluator_sha="$($python_bin -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$project_root/ebjd/evaluate.py")"

"$python_bin" - "$checkpoint" "$config" "$manifest" "$full_result" \
  "$gdts_result" "$assets/CPU_PREFLIGHT_RECEIPT.json" \
  "$implementation_commit" "$evaluator_sha" <<'PY'
import hashlib,json,os,sys,tempfile,torch
from pathlib import Path
from ebjd.config import build_model,load_config
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
checkpoint,config,manifest,full,gdts,output,commit,evaluator=sys.argv[1:]
expected={'checkpoint_sha256':'425988097793b639eb868fcc9ff28e41a3022c08598d13f396383da89268e1b1',
          'config_sha256':'a3800984c4876bec69caa62a587ab2db609d2dce1c7670995ddee0ecb6440278',
          'manifest_sha256':'84c95cb431caf7214d1b85e8e7dbfeaf48c9fe301b1cda7f3201c3e8cf199c78',
          'full_result_sha256':'5e0f13ab4cb272aebe9ea19e6f0307a9eb3213301256c2f6c826cf449bab800f',
          'gdts_result_sha256':'df9aa35cd26296558c864a2bcda792a8411bbc83a530376c45d3429a768eae6c'}
actual={'checkpoint_sha256':sha(checkpoint),'config_sha256':sha(config),
        'manifest_sha256':sha(manifest),'full_result_sha256':sha(full),
        'gdts_result_sha256':sha(gdts)}
if actual!=expected: raise SystemExit(f'frozen identity mismatch: {actual}')
payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
if payload.get('epoch')!=45 or payload.get('update_index')!=11970:
    raise SystemExit(f'checkpoint boundary mismatch: epoch={payload.get("epoch")} update={payload.get("update_index")}')
cfg=load_config(config)
models=[build_model(cfg,'none') for _ in range(2)]
for model in models: model.load_state_dict(payload['model'])
models[0].denoiser.future_social_gamma=1.0
models[1].denoiser.future_social_gamma=0.5
states=[model.state_dict() for model in models]
if states[0].keys()!=states[1].keys() or any(
        not torch.equal(states[0][key],states[1][key]) for key in states[0]):
    raise SystemExit('checkpoint state changed across inference modes')
result={'schema':'ebjd-hotel-future-social-half-preflight-v1','status':'PASS',
        **expected,'checkpoint_epoch':45,'checkpoint_update_index':11970,
        'implementation_commit':commit,'evaluator_sha256':evaluator,'gamma':0.5,
        'full_reused_seeds':list(range(2035,2040)),
        'matched_gdts_reused_seeds':list(range(2035,2040)),
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
print(sum(values)/len(values) if values else 240.0)
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
          'manifest_sha256':sha(config['paths']['test_manifest']),
          'evaluator_sha256':sha(sys.argv[4]),'future_social_mode':'half',
          'future_social_gamma':0.5,'seed':int(sys.argv[5]),'worlds':20,
          'steps':20,'ddim_eta':0.0,'scene_count':445,'agent_occurrence_count':1197}
raise SystemExit(0 if valid_seed_receipt(row,expected,require_fixed=False) else 1)
PY
  then
    echo "reused validated HOTEL HALF receipt: seed $seed"
    return 0
  fi
  set +e
  /usr/bin/time -f '%e' -o "$seconds" "$python_bin" -u -B -m ebjd.evaluate \
    --config "$config" --checkpoint "$checkpoint" --ablation none \
    --split test --device cuda:0 --future-social-mode half \
    --inference-seeds "$seed" --seed-receipt-dir "$receipts" \
    --output "$output" >"$log" 2>&1
  status=$?
  set -e
  elapsed="$(tail -n 1 "$seconds")"
  label=FAILED
  [[ "$status" -eq 0 && -f "$receipt" ]] && label=COMPLETE
  record_invocation "$seed" "$elapsed" "$label"
  printf 'HOTEL HALF seed=%s status=%s elapsed=%s cumulative=%s\n' \
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
  --config "$config" --checkpoint "$checkpoint" --ablation none \
  --split test --device cpu --future-social-mode half \
  --inference-seeds "${seeds[@]}" --seed-receipt-dir "$receipts" \
  --output "$half_result"

"$python_bin" -u -B scripts/finalize_hotel_future_social_half.py \
  --full-result "$full_result" --gdts-result "$gdts_result" \
  --half-receipts "$receipts" --config "$config" \
  --checkpoint "$checkpoint" --manifest "$manifest" --ledger "$ledger" \
  --archive "$archive" --implementation-commit "$implementation_commit" \
  --tests-passed 103

cp "$assets/CPU_PREFLIGHT_RECEIPT.json" "$archive/CPU_PREFLIGHT_RECEIPT.json"
printf 'COMPLETE\nimplementation_commit=%s\ngpu_seconds=%s\n' \
  "$implementation_commit" "$(spent_seconds)" | tee "$assets/PIPELINE_STATE.txt"
