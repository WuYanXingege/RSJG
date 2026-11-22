"""Execute only one authorized ZARA2 dependence model; preserve all older queues."""
import argparse
import fcntl
import hashlib
import os
import subprocess
import time
import torch
from bcep.contracts import canonical_hash, digest
from bcep_joint.common import read, write, seed_all
from bcep_v2.runtime import configure_cuda
from .inputs import Inputs, RUN, ROOT, SOURCE, ARCHIVE, RECIPE, LIMITS, source_hashes
from .model import EdgeDependence
from .cache import build_cache
from .preflight import preflight
from .train import train
from .evaluation import evaluate_final


def state_hash(model):
    return canonical_hash({k: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                           for k, v in model.state_dict().items()})


def wait_gpu():
    while True:
        raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
        pids = [x.strip() for x in raw.splitlines() if x.strip()]
        if not pids:return
        write(RUN/'STATUS.json', dict(status='WAITING_FOR_GPU', compute_pids=pids, unix=time.time()))
        print('WAITING_FOR_GPU', pids, flush=True)
        time.sleep(45)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-gpu', action='store_true')
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN/'STATUS.json').exists() and read(RUN/'STATUS.json')['status'] == 'COMPLETE':
            print(read(RUN/'DECISION.json'), flush=True);return
        torch.set_num_threads(1)
        inputs = Inputs()
        ssec = ROOT/'RSJG_EBJD_assets/experiments/zara2_ssec_energy_stage2_20261121/SUPERVISOR_STATUS.json'
        assert read(ssec)['status'] == 'PAUSED_NO_AUTOMATIC_RESUME'
        protection = dict(ssec_supervisor_sha256=digest(ssec),
                          closed_selector_status_sha256=digest(ROOT/'RSJG_EBJD_assets/experiments/zara2_scene_pair_selector_20261122/STATUS.json'))
        sources = source_hashes()
        config = dict(schema='zara2-continuous-joint-flow-v1', identity=inputs.identity,
            runtime_source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip(),
            reference_archive_commit=ARCHIVE, source_sha256=sources, recipe=RECIPE,
            model='one shared Linear260->64, SiLU, Linear64->4; 2x2 endpoint loadings from both directed viewpoints',
            inputs='frozen Flow context_i/context_j (128 each), observed relative xy (m), relative velocity (m/s)',
            initialization=dict(output_weight_normal_std=.001, output_bias=[.02,0.,0.,.02]),
            graph='undirected union of original observed nearest8 graph, including boundary ties',
            construction='u_i=epsilon_i+sum B_i,e xi_e; L_i=chol(I+sum B_i,e B_i,e^T); z_i=L_i^-1 u_i; R=F F^T',
            loss='[logdet R + zstar^T R^-1 zstar - zstar^T zstar]/(2N), window-equally-weighted nat/person',
            targets='true synchronous endpoint -> original motion_frame -> frozen four-layer Flow inverse',
            precision='FP32 model, NLL, Flow and bridge; float64 CPU discrete medoid distance bookkeeping',
            compression='256 full joint draws -> 20 distinct actual medoids; mean-agent Euclidean metric; stable raw draw ID ties',
            correction='one frozen epoch9 head call after compression', bridge='six layers, eta=0, 20 DDIM steps',
            bypass=['per-agent KMeans', 'Hungarian', 'posthoc selector'],
            streams=['personal Gaussian', 'undirected-edge Gaussian sorted agent IDs', 'bridge keyed by original raw draw ID', 'fixed correspondence shuffles'],
            candidate_seed=2035, evaluation_seed=2035, training_seed=3201, raw_samples=256, K=20,
            selector='minimum all-dev B JADE after real K20/20step decoding each epoch; ties earlier completed epoch',
            thresholds=LIMITS, final_comparisons=['A archived original H0 with native spot replay', 'B rank1 dependence',
                'C independent epsilon, same base draws/compression/head/bridge', 'D five fixed per-agent joint-goal/noise permutations of B'],
            D_repetitions=5, D_aggregation='average native statistics within each window, then native global accumulator',
            theoretical_marginal_claim='each precompression continuous Flow marginal only; finite256/selected20/corrected20 coverage measured separately',
            frozen='Provider density19, correction head9, old coordinate head, shared encoder and bridge eval with requires_grad=False',
            retention='ALL complete epoch snapshots + best + last + Top5 distinct completed epochs ranked only by B dev JADE; no deletion',
            train_windows=2681, dev_windows=590, dev_dense_windows=127, deterministic=True, TF32=False, SDPA='math',
            test_access=False, other_folds=False, other_seeds=False, threshold_search=False, protection=protection)
        if (RUN/'RUN_CONFIG.json').exists():assert read(RUN/'RUN_CONFIG.json') == config, 'source/config mismatch on resume'
        else:write(RUN/'RUN_CONFIG.json', config)
        if args.wait_gpu:wait_gpu()
        else:
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(), 'GPU busy; use --wait-gpu'
        configure_cuda()
        device = torch.device('cuda')
        model, provider = inputs.load_frozen(device)
        provider.to(device).eval().requires_grad_(False)
        seed_all(3201)
        dependence = EdgeDependence().to(device)
        before = dict(backbone=state_hash(model), provider=state_hash(provider))
        write(RUN/'FROZEN_MODELS.json', dict(before=before, provider=inputs.provider_identity))
        try:
            write(RUN/'STATUS.json', dict(status='PREFLIGHT', unix=time.time()))
            preflight(inputs, model, provider, dependence, device)
            write(RUN/'STATUS.json', dict(status='FROZEN_LATENT_CACHE', unix=time.time()))
            cache = build_cache(inputs, provider, device)
            identity = dict(inputs.identity, config=canonical_hash(config),
                            latent_caches={p:digest(RUN/'latent_cache'/f'{p}_MANIFEST.json') for p in cache})
            write(RUN/'STATUS.json', dict(status='TRAINING', unix=time.time()))
            dependence = train(inputs, model, provider, dependence, cache, identity, device)
            write(RUN/'STATUS.json', dict(status='FINAL_VALIDATION', unix=time.time()))
            result = evaluate_final(inputs, model, provider, dependence, cache['selection_dev'], device,
                                    rank1_epoch=read(RUN/'TRAINING_RESULT.json')['rank1_epoch'])
            after = dict(backbone=state_hash(model), provider=state_hash(provider))
            assert before == after
            assert not any(p.requires_grad for p in list(model.parameters())+list(provider.parameters()))
            assert not any(m.training for m in list(model.modules())+list(provider.modules()))
            assert source_hashes() == sources
            assert digest(ssec) == protection['ssec_supervisor_sha256']
            assert digest(ROOT/'RSJG_EBJD_assets/experiments/zara2_scene_pair_selector_20261122/STATUS.json') == protection['closed_selector_status_sha256']
            write(RUN/'FROZEN_MODELS.json', dict(before=before, after=after, exact_unchanged=True, provider=inputs.provider_identity))
            write(RUN/'STATUS.json', dict(status='COMPLETE', result=result, unix=time.time()))
            print(result, flush=True)
        except BaseException as error:
            write(RUN/'STATUS.json', dict(status='INTERRUPTED_RESUME_SAME_LAST', error=repr(error), unix=time.time()))
            raise


if __name__ == '__main__':main()
