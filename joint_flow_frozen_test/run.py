"""Export fixed A/B trajectories, seal all five seeds, then score heldout truth."""
import argparse
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import subprocess
import time

import numpy as np
import torch

from bcep.contracts import canonical_hash, digest
from bcep_formal.data import FormalData
from bcep_joint.common import read, write
from bcep_joint.inference import generate as generate_a
from bcep_v2.bank import export
from bcep_v2.runtime import configure_cuda
from ebjd.metrics import MetricAccumulator, hard_metric_statistics
from joint_flow_dependency.inference import generate as generate_b
from joint_flow_dependency.inputs import source_hashes as dependency_sources
from .inputs import Inputs, SOURCE, ROOT, RUN, BASE_COMMIT, SEEDS, PURPOSE, B_SHA
from .scoring import score


def source_hashes():
    return dict(dependency_sources(), **{str(p.relative_to(SOURCE)): digest(p)
                for p in sorted((SOURCE/'joint_flow_frozen_test').rglob('*.py'))})


def immutable(path, value):
    if path.exists():
        assert read(path) == value, 'immutable run identity mismatch: '+path.name
    else:
        write(path, value)


def state_hash(module):
    return canonical_hash({key: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                           for key, value in module.state_dict().items()})


def csvwrite(path, rows):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


@contextmanager
def forbid_heldout_truth():
    original = FormalData.item
    def guarded(self, index, purpose, truth=True):
        if purpose == PURPOSE and truth:
            raise RuntimeError('heldout truth forbidden until all A/B five-seed predictions are sealed')
        return original(self, index, purpose, truth)
    FormalData.item = guarded
    try:
        yield
    finally:
        FormalData.item = original


def progress(phase, position, total, start, **extra):
    if position % 25 == 0 or position == total:
        row = dict(phase=phase, windows=position, total=total,
                   seconds=time.monotonic()-start, unix=time.time(), **extra)
        write(RUN/'PROGRESS.json', row)
        print(row, flush=True)


def wait_gpu():
    while True:
        pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                      '--format=csv,noheader,nounits'], text=True).strip()
        if not pids:
            return
        write(RUN/'STATUS.json', dict(status='WAITING_FOR_GPU', compute_pids=pids.splitlines(), unix=time.time()))
        print('WAITING_FOR_GPU', pids.splitlines(), flush=True)
        time.sleep(45)


@torch.no_grad()
def replay_dev(inputs, model, provider, dependence, device):
    receipt = RUN/'DEV_REPLAY.json'
    config_sha = digest(RUN/'RUN_CONFIG.json')
    if receipt.exists():
        saved = read(receipt)
        assert saved['passed'] and saved['exact_metrics']
        assert saved['source_config_sha256'] == config_sha
        assert saved['per_window_metrics_sha256'] == digest(RUN/'DEV_REPLAY.csv')
        return saved
    ids = inputs.split['selection_dev_indices']
    bank_replay = []
    checks = [ids[0], ids[1], next(i for i in ids if len(inputs.split['members'][i]['agent_ids']) >= 9)]
    for index in checks:
        host = inputs.dev_data.batch([index], 'selection_dev', truth=False)
        batch = host.to(device)
        expected = inputs.dev_bank2035.batch([index], host).to(device)[0]
        actual = provider.candidates(batch.observed[0], batch.semantic_maps[0], batch.valid[0],
            batch.scene_ids[0], batch.agent_ids[0], 2035, inputs.geometry, refined=True)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        bank_replay.append(dict(window=f'dev_{ids.index(index)+1:04d}',
                               max_abs_m=float((actual-expected).abs().max()), exact=torch.equal(actual, expected)))
    accumulators = {branch: {group: MetricAccumulator() for group in ('all', 'N_ge_9')}
                    for branch in ('A', 'B')}
    rows = []
    start = time.monotonic()
    for position, index in enumerate(ids, 1):
        host = inputs.dev_data.batch([index], 'selection_dev', truth=False)
        batch = host.to(device)
        assert not batch.future.count_nonzero()
        goals = inputs.dev_bank2035.batch([index], host).to(device)
        _, a, _, _ = generate_a(model, batch.observed, batch.semantic_maps, batch.valid,
                                batch.scene_ids, batch.agent_ids, goals, 2035, mode='H')
        b, _, detail = generate_b(model, provider, dependence, batch, seed=2035)
        assert detail['ddim_steps'] == 20 and detail['ddim_calls'] == 1
        # This is dev truth only; no heldout truth access is allowed in this phase.
        truth = inputs.dev_data.batch([index], 'selection_dev', truth=True).future.to(device)
        row = dict(window=f'dev_{position:04d}', agents=int(batch.valid.sum()))
        for branch, prediction in (('A', a), ('B', b)):
            statistics = hard_metric_statistics(prediction, truth, batch.valid)
            for group in ('all', 'N_ge_9'):
                if group == 'all' or row['agents'] >= 9:
                    accumulators[branch][group].update_statistics(statistics)
            row.update({branch+'_JADE': float(statistics.scene_jade[0]),
                        branch+'_JFDE': float(statistics.scene_jfde[0]),
                        branch+'_minADE': float(statistics.agent_minade.double().mean()),
                        branch+'_minFDE': float(statistics.agent_minfde.double().mean())})
        rows.append(row)
        progress('DEV_SEED2035_REPLAY', position, len(ids), start)
    metrics = {branch: {group: acc.as_dict() for group, acc in groups.items()}
               for branch, groups in accumulators.items()}
    reference = {branch: inputs.archive_metrics[branch] for branch in ('A', 'B')}
    assert metrics == reference, 'seed2035 dev replay differs from the fixed archived A/B metrics'
    csvwrite(RUN/'DEV_REPLAY.csv', rows)
    result = dict(passed=True, seed=2035, windows=590, dense_windows=127,
                  reference_commit=BASE_COMMIT, metrics=metrics, reference=reference,
                  exact_metrics=True, per_window_metrics_sha256=digest(RUN/'DEV_REPLAY.csv'),
                  source_config_sha256=config_sha, native_provider_bank_replay=bank_replay,
                  heldout_future_reads=0, unix=time.time())
    write(receipt, result)
    return result


@torch.no_grad()
def ensure_a_bank(inputs, provider, seed, device):
    if seed in inputs.test_banks:
        return inputs.test_banks[seed]
    folder = RUN/'A_refined_banks'/f'{PURPOSE}_{seed}'
    start = time.monotonic()
    ids = inputs.split[PURPOSE+'_indices']
    positions = {index: position for position, index in enumerate(ids, 1)}
    def predict(index):
        batch = inputs.export_data.batch([index], PURPOSE, truth=False).to(device)
        assert not batch.future.count_nonzero()
        goals = provider.candidates(batch.observed[0], batch.semantic_maps[0], batch.valid[0],
            batch.scene_ids[0], batch.agent_ids[0], seed, inputs.geometry, refined=True)
        progress('A_NATIVE_CANDIDATE_EXPORT', positions[index], len(ids), start, seed=seed)
        return goals
    export(folder, inputs.bank_identities[seed], inputs.split, predict)
    return inputs.attach_test_bank(seed, folder)


@torch.no_grad()
def export_predictions(inputs, model, provider, dependence, device):
    target = RUN/'PREDICTIONS_FROZEN.json'
    config_sha = digest(RUN/'RUN_CONFIG.json')
    if target.exists():
        sealed = read(target)
        assert sealed['identity'] == inputs.identity and sealed['source_config_sha256'] == config_sha
        return sealed  # scorer verifies every committed trajectory before truth access
    rows = []
    ids = inputs.split[PURPOSE+'_indices']
    case_hash = canonical_hash(dict(identity=inputs.identity, source_config_sha256=config_sha))
    bank_hashes = {}
    for seed in SEEDS:
        write(RUN/'STATUS.json', dict(status='OBSERVED_ONLY_A_CANDIDATES', seed=seed, unix=time.time()))
        bank = ensure_a_bank(inputs, provider, seed, device)
        bank_hashes[str(seed)] = bank.hash
        write(RUN/'STATUS.json', dict(status='OBSERVED_ONLY_AB_PREDICTIONS', seed=seed, unix=time.time()))
        folder = RUN/'predictions'/f'seed_{seed}'
        folder.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        for position, index in enumerate(ids, 1):
            n = len(inputs.split['members'][index]['agent_ids'])
            path = folder/f'{index}.npz'
            side = path.with_suffix('.json')
            expected = dict(seed=seed, index=index, window=f'test_{position:04d}', n=n,
                            file=str(path.relative_to(RUN)), identity=case_hash, A_bank_sha256=bank.hash)
            if side.exists():
                saved = read(side)
                assert saved == dict(expected, sha256=digest(path)), 'changed or misaligned prediction'
            else:
                host = inputs.export_data.batch([index], PURPOSE, truth=False)
                batch = host.to(device)
                assert batch.valid.all() and not batch.future.count_nonzero()
                goals = bank.batch([index], host).to(device)
                _, a, permutation, a_proof = generate_a(model, batch.observed, batch.semantic_maps,
                    batch.valid, batch.scene_ids, batch.agent_ids, goals, seed, mode='H')
                b, endpoints, detail = generate_b(model, provider, dependence, batch, seed=seed)
                assert a_proof['exact_noise_gather'] and a_proof['per_agent_bijection']
                assert detail['ddim_steps'] == 20 and detail['ddim_calls'] == detail['correction_calls'] == 1
                assert len(endpoints['raw_indices'].unique()) == 20
                arrays = dict(A=a[0].cpu().numpy(), B=b[0].cpu().numpy())
                assert all(v.shape == (20,n,12,2) and v.dtype == np.float32 and np.isfinite(v).all()
                           for v in arrays.values())
                if path.exists():
                    with np.load(path, allow_pickle=False) as orphan:
                        assert set(orphan.files) == {'A', 'B'}
                        assert all(np.array_equal(orphan[key], value) for key, value in arrays.items())
                else:
                    temporary = path.with_suffix('.tmp.npz')
                    np.savez_compressed(temporary, **arrays)
                    temporary.replace(path)
                saved = dict(expected, sha256=digest(path))
                write(side, saved)
            rows.append(saved)
            progress('OBSERVED_ONLY_AB_PREDICTIONS', position, len(ids), start, seed=seed)
    assert len(rows) == 998*5
    sealed = dict(schema='zara2-frozen-ab-five-seed-predictions-v1', identity=inputs.identity,
        source_config_sha256=config_sha, windows=998, seeds=list(SEEDS), K=20, M=256,
        ddim_steps=20, prediction_frames=12, prediction_dtype='float32',
        A_bank_sha256=bank_hashes, heldout_future_reads=0, rows=rows, seal_unix=time.time())
    immutable(target, sealed)
    write(RUN/'A_BANKS.json', {str(seed): dict(seed=seed, bank_sha256=bank.hash,
        complete_manifest_sha256=digest(bank.folder/'COMPLETE.json'), windows=len(bank.rows),
        source='original_formal_bank' if not bank.folder.is_relative_to(RUN) else 'this_frozen_test_export')
        for seed, bank in inputs.test_banks.items()})
    return sealed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-gpu', action='store_true')
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN/'STATUS.json').exists() and read(RUN/'STATUS.json')['status'] == 'COMPLETE':
            print(read(RUN/'SUMMARY.json'), flush=True)
            return
        torch.set_num_threads(1)
        inputs = Inputs()
        protected = {
            'SSEC_supervisor': ROOT/'RSJG_EBJD_assets/experiments/zara2_ssec_energy_stage2_20261121/SUPERVISOR_STATUS.json',
            'closed_dependency_run': ROOT/'RSJG_EBJD_assets/experiments/zara2_joint_flow_dependency_20261122/STATUS.json',
            'closed_selector_run': ROOT/'RSJG_EBJD_assets/experiments/zara2_scene_pair_selector_20261122/STATUS.json',
        }
        assert read(protected['SSEC_supervisor'])['status'] == 'PAUSED_NO_AUTOMATIC_RESUME'
        sources = source_hashes()
        config = dict(schema='zara2-frozen-test-v1', base_commit=BASE_COMMIT,
            runtime_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
            source_sha256=sources, asset_identity=inputs.identity, seeds=list(SEEDS), fold='zara2', purpose=PURPOSE,
            A='native coordinate epoch5 H0 with native frozen Provider M256/KMeans20/refined head9',
            B='frozen continuous joint Flow dependence epoch35; whole-scene 256->20 medoids and frozen head9',
            B_checkpoint_sha256=B_SHA, M=256, K=20, ddim_steps=20, prediction_frames=12, eta=0.,
            precision='FP32 models/trajectories/native hard metrics; original float64 discrete selection accounting',
            test_scoring='native CPU hard_metric_statistics and MetricAccumulator, matching formal frozen test protocol',
            metrics=['JADE','JFDE','minADE','minFDE'], groups=['all','N_ge_9'],
            joint_aggregation='scene macro', marginal_aggregation='valid-agent micro',
            seed_aggregation='five independent K20 scores; mean and sample standard deviation ddof=1; paired B-A per seed',
            heldout_windows=998,
            heldout_dense_windows=sum(len(inputs.split['members'][i]['agent_ids'])>=9 for i in inputs.split[PURPOSE+'_indices']),
            prediction_barrier='all A/B trajectories for all 998 windows and all five seeds saved and SHA256-sealed before any heldout future read',
            dev_replay='seed2035, full590 A/B real inference, exact native metrics vs fixed d4fbe6d archive',
            seed_plumbing='explicit generate->sample_endpoints->personal/edge Gaussian and raw-ID bridge noise',
            no_training=True, optimizer_updates=0, hyperparameters_changed=False, model_selection_changed=False,
            additional_folds=False, additional_seeds=False, test_driven_selection=False,
            protection={key:digest(path) for key,path in protected.items()})
        immutable(RUN/'RUN_CONFIG.json', config)
        if args.wait_gpu:
            wait_gpu()
        else:
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                '--format=csv,noheader,nounits'],text=True).strip(), 'GPU busy; use --wait-gpu'
        configure_cuda()
        device = torch.device('cuda')
        model, provider, dependence = inputs.load_frozen(device)
        modules = dict(A=model, Provider=provider, B=dependence)
        before = {key:state_hash(module) for key,module in modules.items()}
        write(RUN/'FROZEN_MODELS.json', dict(before=before, optimizer_updates=0))
        try:
            with forbid_heldout_truth():
                write(RUN/'STATUS.json', dict(status='DEV_SEED2035_REPLAY', unix=time.time()))
                replay_dev(inputs, model, provider, dependence, device)
                sealed = export_predictions(inputs, model, provider, dependence, device)
            after = {key:state_hash(module) for key,module in modules.items()}
            assert before == after and source_hashes() == sources
            assert not any(p.requires_grad for module in modules.values() for p in module.parameters())
            assert not any(m.training for module in modules.values() for m in module.modules())
            for module in modules.values():
                module.cpu()
            torch.cuda.empty_cache()
            write(RUN/'FROZEN_MODELS.json', dict(before=before, after=after, exact_unchanged=True,
                                               requires_grad_false=True, optimizer_updates=0))
            write(RUN/'STATUS.json', dict(status='SCORING_SEALED_PREDICTIONS', unix=time.time()))
            result = score(inputs, sealed)
            assert source_hashes() == sources
            assert all(digest(path) == config['protection'][key] for key,path in protected.items())
            write(RUN/'PROTECTED_RUNS.json', dict(unchanged=True, before=config['protection'],
                after={key:digest(path) for key,path in protected.items()}, SSEC_still_paused=True))
            write(RUN/'STATUS.json', dict(status='COMPLETE', score_receipt=result, unix=time.time()))
            print(read(RUN/'SUMMARY.json'), flush=True)
        except BaseException as error:
            write(RUN/'STATUS.json', dict(status='INTERRUPTED_RESUME_SAME_FROZEN_TEST', error=repr(error), unix=time.time()))
            raise


if __name__ == '__main__':
    main()
