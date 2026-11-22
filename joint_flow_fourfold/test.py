"""Actual A/B five-seed inference, per-fold global sealing, then native scoring."""
from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import torch

from bcep.contracts import canonical_hash, digest
from bcep_joint.common import read, write
from bcep_joint.inference import generate as generate_a
from bcep_v2.bank import Bank, export
from joint_flow_dependency.inference import generate as generate_b
from joint_flow_dependency.model import EdgeDependence
from joint_flow_frozen_test.inputs import ObservedOnlyHeldoutData
from joint_flow_frozen_test.run import forbid_heldout_truth, state_hash
from .common import SEEDS, PURPOSE, immutable, status
from .inputs import Inputs
from .scoring import score_fold


def progress(output, phase, position, total, start, **extra):
    if position % 25 == 0 or position == total:
        row = dict(phase=phase, windows=position, total=total,
                   seconds=time.monotonic()-start, unix=time.time(), **extra)
        write(output/'PROGRESS.json', row)
        print(row, flush=True)


@torch.no_grad()
def run_test(upstream, selected, barrier, output, device='cuda'):
    """Only a manifest admitted by the cross-fold freeze barrier can be tested."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fold = upstream['fold']
    assert barrier['status'] == 'ALL_AVAILABLE_MODELS_FROZEN_BEFORE_TEST'
    admitted = barrier['folds'][fold]
    assert admitted['A_sha256'] == upstream['joint']['sha256']
    assert admitted['B_sha256'] == selected['rank1_sha256']
    assert digest(selected['rank1_checkpoint']) == selected['rank1_sha256']
    inputs = Inputs(upstream, output/'input_validation', device)
    split = inputs.split
    ids = split[PURPOSE+'_indices']
    assert ids and fold in barrier['requested_folds']
    identity = dict(upstream=inputs.identity, B_sha256=selected['rank1_sha256'],
        B_epoch=selected['rank1_epoch'], freeze_barrier=canonical_hash(barrier), fold=fold,
        heldout_members=canonical_hash([split['members'][i] for i in ids]))
    config = dict(fold=fold, identity=identity, seeds=list(SEEDS), M=256, K=20,
        ddim_steps=20, eta=0., prediction_frames=12, windows=len(ids),
        A='native coordinate H0 dev rank1', B='frozen dependency dev JADE rank1',
        scoring='native CPU FP32 hard metrics; FP64 native accumulation',
        prediction_barrier='all five seeds and A/B saved and sealed before heldout truth',
        joint_aggregation='equal windows', marginal_aggregation='valid-agent weighted',
        no_heldout_model_selection=True, frozen_cross_fold_manifest=canonical_hash(barrier))
    immutable(output/'RUN_CONFIG.json', config)
    observed = ObservedOnlyHeldoutData(split)
    model, provider = inputs.load_frozen(device)
    saved = torch.load(selected['rank1_checkpoint'], map_location='cpu', weights_only=False)
    assert saved['completed_epoch'] == selected['rank1_epoch']
    dependence = EdgeDependence()
    dependence.load_state_dict(saved['model'], strict=True)
    for module in (model, provider, dependence):
        module.to(device).eval().requires_grad_(False)
    modules = dict(A=model, Provider=provider, B=dependence)
    before = {k: state_hash(v) for k, v in modules.items()}
    write(output/'FROZEN_MODELS.json', dict(before=before, optimizer_updates=0))
    template = read(inputs.banks['selection_dev'].folder/'IDENTITY.json')
    bank_hashes = {}
    target = output/'PREDICTIONS_FROZEN.json'
    with forbid_heldout_truth():
        if target.exists():
            seal = read(target)
            assert seal['identity'] == identity
        else:
            rows = []
            case = canonical_hash(identity)
            for seed in SEEDS:
                status(output, 'OBSERVED_ONLY_A_CANDIDATES', fold=fold, seed=seed)
                bank_id = deepcopy(template)
                bank_id.update(purpose=PURPOSE, seed=seed, member_hash=identity['heldout_members'])
                bank_folder = output/'A_refined_banks'/f'{PURPOSE}_{seed}'
                start = time.monotonic()
                positions = {index: pos for pos, index in enumerate(ids, 1)}
                def predict(index):
                    batch = observed.batch([index], PURPOSE, truth=False).to(device)
                    assert not batch.future.count_nonzero()
                    goals = provider.candidates(batch.observed[0], batch.semantic_maps[0], batch.valid[0],
                        batch.scene_ids[0], batch.agent_ids[0], seed, inputs.geometry, refined=True)
                    progress(output, 'A_NATIVE_CANDIDATES', positions[index], len(ids), start, seed=seed)
                    return goals
                export(bank_folder, bank_id, split, predict)
                bank = Bank(bank_folder, bank_id, split)
                bank_hashes[str(seed)] = bank.hash
                folder = output/'predictions'/f'seed_{seed}'
                folder.mkdir(parents=True, exist_ok=True)
                status(output, 'OBSERVED_ONLY_AB_PREDICTIONS', fold=fold, seed=seed)
                start = time.monotonic()
                for pos, index in enumerate(ids, 1):
                    n = len(split['members'][index]['agent_ids'])
                    path = folder/f'{index}.npz'
                    side = path.with_suffix('.json')
                    expected = dict(seed=seed, index=index, window=f'test_{pos:04d}', n=n,
                        file=str(path.relative_to(output)), identity=case, A_bank_sha256=bank.hash)
                    if side.exists():
                        row = read(side)
                        assert row == dict(expected, sha256=digest(path)), 'prediction changed'
                    else:
                        host = observed.batch([index], PURPOSE, truth=False)
                        batch = host.to(device)
                        assert batch.valid.all() and not batch.future.count_nonzero()
                        goals = bank.batch([index], host).to(device)
                        _, a, _, proof = generate_a(model, batch.observed, batch.semantic_maps,
                            batch.valid, batch.scene_ids, batch.agent_ids, goals, seed, mode='H')
                        b, endpoints, detail = generate_b(model, provider, dependence, batch, seed=seed)
                        assert proof['exact_noise_gather'] and proof['per_agent_bijection']
                        assert detail['ddim_steps'] == 20 and detail['ddim_calls'] == detail['correction_calls'] == 1
                        assert len(endpoints['raw_indices'].unique()) == 20
                        arrays = dict(A=a[0].cpu().numpy(), B=b[0].cpu().numpy())
                        assert all(x.shape == (20,n,12,2) and x.dtype == np.float32 and np.isfinite(x).all()
                                   for x in arrays.values())
                        if path.exists():
                            with np.load(path, allow_pickle=False) as prior:
                                assert set(prior.files) == {'A', 'B'}
                                assert all(np.array_equal(prior[k], v) for k, v in arrays.items())
                        else:
                            temporary = path.with_suffix('.tmp.npz')
                            np.savez_compressed(temporary, **arrays)
                            temporary.replace(path)
                        row = dict(expected, sha256=digest(path))
                        write(side, row)
                    rows.append(row)
                    progress(output, 'OBSERVED_ONLY_AB_PREDICTIONS', pos, len(ids), start, seed=seed)
            assert len(rows) == len(ids)*len(SEEDS)
            seal = dict(schema='fourfold-frozen-ab-five-seed-predictions-v1', fold=fold,
                identity=identity, source_config_sha256=digest(output/'RUN_CONFIG.json'),
                windows=len(ids), seeds=list(SEEDS), K=20, M=256, ddim_steps=20,
                prediction_frames=12, prediction_dtype='float32', A_bank_sha256=bank_hashes,
                heldout_future_reads=0, rows=rows, seal_unix=time.time())
            immutable(target, seal)
    after = {k: state_hash(v) for k, v in modules.items()}
    assert before == after
    assert all(not p.requires_grad for m in modules.values() for p in m.parameters())
    for module in modules.values():
        module.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    write(output/'FROZEN_MODELS.json', dict(before=before, after=after, exact_unchanged=True, optimizer_updates=0))
    status(output, 'SCORING_SEALED_PREDICTIONS', fold=fold)
    result = score_fold(split, seal, output)
    status(output, 'COMPLETE', fold=fold, score_receipt=result)
    return result
