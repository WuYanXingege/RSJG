"""Execute the authorized single trial; wait for existing GPU work to finish."""
import argparse
import fcntl
import os
import subprocess
import time
import traceback
import torch
from bcep.contracts import canonical_hash, digest
from bcep_joint.common import read, write
from bcep_v2.runtime import configure_cuda
from .inputs import Inputs, RUN, SOURCE, ROOT, RECIPE, LIMITS, source_hashes
from .model import SceneScorer
from .cache import preflight, build_cache, load_cache, accumulate
from .train import train
from .finalize import evaluate


def state_hash(model):
    import hashlib
    return canonical_hash({k: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                           for k, v in model.state_dict().items()})


def wait_for_gpu():
    status = ROOT/'RSJG_EBJD_assets/experiments/zara2_ssec_energy_stage2_20261121/STATUS.json'
    while True:
        raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
        active = [x.strip() for x in raw.splitlines() if x.strip()]
        old_status = read(status)['status']
        if not active and old_status == 'COMPLETE':
            write(RUN/'GPU_RELEASE.json', dict(unix=time.time(), ssec_status=old_status, compute_pids=[]))
            return
        write(RUN/'STATUS.json', dict(status='WAITING_FOR_GPU', ssec_status=old_status, compute_pids=active, unix=time.time()))
        print('WAITING_FOR_GPU', old_status, active, flush=True)
        time.sleep(45)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-gpu', action='store_true')
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN/'STATUS.json').exists() and read(RUN/'STATUS.json')['status'] == 'COMPLETE':
            print(read(RUN/'DECISION.json'), flush=True)
            return
        torch.set_num_threads(1)
        inputs = Inputs()
        source = source_hashes()
        config = dict(schema='zara2-scene-pair-selector-v1', identity=inputs.identity,
            runtime_source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip(),
            source_sha256=source, recipe=RECIPE, normalization='fit valid train features only, std floor 1e-3',
            width=32, selector_parameters=sum(p.numel() for p in SceneScorer().parameters()),
            node_inputs='history relative to last position, last-4 velocity, frozen social_h, candidate displacement',
            edge_inputs='same-world neighbour relative observed position, velocity and paired endpoint; nearest 8 including ties',
            pooling='neighbour mean/max -> agent mean/max -> world MLP -> 20-world mean/max',
            target='[JADE(I)-JADE(H0), JFDE(I)-JFDE(H0)] metres',
            objective='two-output unscaled MSE; each window equal; 16 windows/update, actual tail-size divisor',
            gate='dA_hat <= 0 and dF_hat < 0; singleton/identical always H0',
            selector='minimum all-dev selected JADE; exact ties earlier completed epoch',
            thresholds=LIMITS, candidate_seed=2035, training_seed=3201, precision='FP32',
            deterministic=True, TF32=False, SDPA='math', K=20, DDIM_steps=20,
            retention='ALL complete epoch snapshots + best + last + Top5 distinct complete epochs ranked only by selected dev JADE; no deletion',
            frozen='Provider, coordinate pairing head, shared encoder, six-layer bridge: eval and requires_grad=False',
            optimization='only SceneScorer parameters; native AdamW decay groups, 5-epoch warmup and horizon100 cosine',
            train_windows=2681, dev_windows=590, dev_dense_windows=127,
            test_access=False, threshold_search=False, other_seeds=False, other_folds=False)
        path = RUN/'RUN_CONFIG.json'
        if path.exists():
            assert read(path) == config, 'resume must use the identical source and configuration'
        else:
            write(path, config)
        if args.wait_gpu:
            wait_for_gpu()
        configure_cuda()
        device = torch.device('cuda')
        model, provider = inputs.load_frozen(device)
        before = dict(backbone=state_hash(model), provider=state_hash(provider))
        write(RUN/'FROZEN_MODELS.json', dict(before=before, provider=inputs.provider_identity))
        try:
            write(RUN/'STATUS.json', dict(status='PREFLIGHT', unix=time.time()))
            preflight(inputs, model, device)
            assert read(RUN/'PREFLIGHT.json')['passed']
            write(RUN/'STATUS.json', dict(status='LABELS', unix=time.time()))
            records = build_cache(inputs, model, device)
            write(RUN/'CACHE_BASELINES.json', dict(always_H0=accumulate(records['selection_dev']),
                  always_I=accumulate(records['selection_dev'], [True]*590)))
            ident = dict(inputs.identity, config=canonical_hash(config),
                         caches={p: digest(RUN/'cache'/f'{p}_MANIFEST.json') for p in records})
            write(RUN/'STATUS.json', dict(status='TRAINING', unix=time.time()))
            selector = train(records, ident, device)
            write(RUN/'STATUS.json', dict(status='FINAL_REPLAY', unix=time.time()))
            result = evaluate(inputs, model, selector, records['selection_dev'], device)
            after = dict(backbone=state_hash(model), provider=state_hash(provider))
            assert before == after
            assert not any(p.requires_grad for p in model.parameters())
            assert not any(m.training for m in model.modules())
            assert source_hashes() == source
            write(RUN/'FROZEN_MODELS.json', dict(before=before, after=after, exact_unchanged=True,
                                                provider=inputs.provider_identity))
            write(RUN/'STATUS.json', dict(status='COMPLETE', result=result, unix=time.time()))
            print(result, flush=True)
        except BaseException as error:
            write(RUN/'STATUS.json', dict(status='INTERRUPTED_RESUME_SAME_LAST', error=repr(error), unix=time.time()))
            raise


if __name__ == '__main__':
    main()
