"""Seal frozen endpoints, score sealed predictions, and render anonymous cases."""

import argparse
from collections import Counter
import fcntl
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from bcep.contracts import require
from bcep_formal.data import FormalData
from bcep_v2.runtime import configure_cuda
from direct_k20.common import ASSETS, ATOL, ENVIRONMENT, METRICS, RTOL, SOURCE, digest, immutable, read, status, write
from direct_k20.interaction_eval import DEV, TEST, GROUPS, _group, _group_row, _output, _row_metrics
from ebjd.metrics import MetricAccumulator, hard_metric_statistics
from joint_flow_fourfold.report import csvwrite, read_csv
from joint_flow_frozen_test.run import forbid_heldout_truth

from .assets import (
    ARCHIVE, FOLDS, MODELS, ORIGINAL, PURPOSES, frozen_assets, native_endpoints,
    read_arrays, reference_fold, replay_missing, save_npz,
)


BASE = 'efcacc8a7d369c4e6e52ec0d110289b59459e56f'
RUN = ASSETS / 'endpoint_diagnosis_20261127'
DOC = ARCHIVE.parent / 'endpoint_diagnosis_20261127'
BRANCH = 'experiment/hotel-univ-endpoint-diagnosis-20261127'
ENDPOINT_VECTORS = ('raw', 'g0', 'g1', 'unary_world', 'social_world')
NUMERIC_TOL_M = 1e-6


def _make_references():
    return {(purpose, fold): reference_fold(fold, purpose, RUN)
            for purpose in PURPOSES for fold in FOLDS}


def _code_hashes():
    return {path.name: digest(path) for path in sorted(Path(__file__).parent.glob('*.py'))}


def _prepare(refs):
    config = dict(
        base_commit=BASE, purpose='frozen endpoint and trajectory diagnosis',
        sources_sha256=_code_hashes(), folds=list(FOLDS), inference_seeds=[2035, 2036, 2037, 2038, 2039],
        development_seeds=[2035], endpoint_seed=2035, K=20, observed_frames=8, prediction_frames=12,
        ddim_steps=20, eta=0., dt=.4, training_updates=0,
        tolerance=dict(native_coordinate_atol_m=ATOL, native_coordinate_rtol=RTOL,
                       score_replay_abs_m=NUMERIC_TOL_M, bound_fp64_abs_m=1e-12),
        folds_by_purpose={
            f'{purpose}/{fold}': dict(
                **ref.source_hashes, windows=len(ref.indices), seeds=ref.seeds,
                S0_epoch=ref.identity['S0_epoch'], S0_sha256=ref.identity['S0_sha256'],
                R12_epoch=12, R12_sha256=ref.identity['R12_sha256'], split_sha256=ref.identity['split_sha256'])
            for (purpose, fold), ref in refs.items()},
        group_partition='greedy increasing mean distance over last three observed positions, <=2 m',
        random_partition_draws=20, random_partition_seed=3201,
        directional_near_stationary_mps=.2,
        normalize='J/L/G use window means and minFDE uses people',
        environment=ENVIRONMENT, commands=['python -m direct_k20.interaction_diagnosis.run export',
            'python -m direct_k20.interaction_diagnosis.run score',
            'python -m direct_k20.interaction_diagnosis.run publish'],
    )
    immutable(RUN / 'RUN_CONFIG.json', config)
    return config


def _side_identity(ref, ordinal, index):
    original = ref.rows[index, 2035]
    return dict(window=original['window'], n=original['n'], member_sha256=original['member_sha256'],
                file=f'{ordinal:04d}.npz', source_pair_seal_sha256=ref.source_hashes['pair_seal_sha256'],
                run_config_sha256=digest(RUN / 'RUN_CONFIG.json'))


def _endpoint_directory(ref):
    return RUN / 'endpoints' / ref.purpose / ref.fold


def _validated_endpoints(folder, row, expected):
    require(all(row[key] == value for key, value in expected.items()), 'sealed endpoint identity changed')
    path = folder / expected['file']
    require(digest(path) == row['sha256'], 'sealed endpoint file changed')
    shapes = dict.fromkeys(ENDPOINT_VECTORS, (20, row['n'], 2))
    return read_arrays(path, shapes)


def _check_anchors(arrays, pair, first_endpoints):
    anchor_diff = 0.
    seed_diff = 0.
    for model, name in zip(MODELS, ('g0', 'g1')):
        last = pair[model][..., -1, :]
        gold = arrays[name]
        require(np.allclose(last, gold, atol=ATOL, rtol=RTOL),
                f'{model} sealed terminal frame differs from native endpoint')
        anchor_diff = max(anchor_diff, float(np.max(np.abs(last.astype(np.float64)-gold))))
        if model in first_endpoints:
            require(np.allclose(last, first_endpoints[model], atol=ATOL, rtol=RTOL),
                    f'{model} terminal frame differs across inference seeds')
            seed_diff = max(seed_diff, float(np.max(np.abs(last.astype(np.float64)-first_endpoints[model]))))
        else:
            first_endpoints[model] = last.copy()
    return anchor_diff, seed_diff


def _export_fold(ref, assets, student, module):
    folder = _endpoint_directory(ref)
    folder.mkdir(parents=True, exist_ok=True)
    reader = FormalData(ref.split, phase='export', verify=False)
    rows = []
    with forbid_heldout_truth():
        for ordinal, index in enumerate(ref.indices, 1):
            expected = _side_identity(ref, ordinal, index)
            path = folder / expected['file']
            side = path.with_suffix('.json')
            if side.is_file():
                row = read(side)
                _validated_endpoints(folder, row, expected)
                rows.append(row)
                continue
            record = ref.split['records'][index]
            require(digest(record['path']) == record['sha256'], 'locked observed record changed')
            batch = reader.batch([index], ref.purpose, truth=False).to(assets.device)
            require(batch.valid.shape == (1, expected['n']), 'packed observations differ')
            arrays, decomposition = native_endpoints(assets, student, module, batch)
            anchor = seed_drift = 0.
            first = {}
            recovered = []
            for seed in ref.seeds:
                pair, repair = ref.pair(index, seed, reproduce=lambda model, keyed_seed, key:
                                        replay_missing(assets, batch, arrays, model, keyed_seed, key))
                replay_diff, cross_seed = _check_anchors(arrays, pair, first)
                anchor = max(anchor, replay_diff)
                seed_drift = max(seed_drift, cross_seed)
                recovered.extend(repair)
            if path.is_file():
                orphan = read_arrays(path, dict.fromkeys(ENDPOINT_VECTORS, (20, expected['n'], 2)))
                require(all(np.array_equal(orphan[name], arrays[name]) for name in ENDPOINT_VECTORS),
                        'orphaned endpoint shard changed on replay')
            else:
                save_npz(path, **arrays)
            row = expected | dict(sha256=digest(path), decomposition_max_abs_m=decomposition,
                endpoint_replay_max_abs_m=anchor, cross_seed_endpoint_max_abs_m=seed_drift,
                recovered=recovered)
            immutable(side, row)
            rows.append(row)
            if ordinal % 100 == 0 or ordinal == len(ref.indices):
                status(RUN, 'EXPORTING_OBSERVED_ENDPOINTS', fold=ref.fold, split=ref.purpose,
                       windows=ordinal, total=len(ref.indices))
    seal = folder / 'ENDPOINTS_SEALED.json'
    immutable(seal, dict(windows=len(rows), rows=rows, truth_reads=0, stage='OBSERVED_ONLY'))
    return dict(endpoint_seal_sha256=digest(seal), windows=len(rows),
        new_ddim_calls=sum(len(row['recovered']) for row in rows),
        **{key: max(row[key] for row in rows) for key in
           ('endpoint_replay_max_abs_m', 'cross_seed_endpoint_max_abs_m', 'decomposition_max_abs_m')})


def export(refs, device):
    conf = _prepare(refs)
    start = time.time()
    if (RUN / 'ENDPOINTS_EXPORTED.json').is_file():
        previous = read(RUN / 'ENDPOINTS_EXPORTED.json')
        require(previous['config_sha256'] == digest(RUN / 'RUN_CONFIG.json'), 'export identity changed')
        return
    if not (RUN / 'START.json').exists():
        write(RUN / 'START.json', dict(unix=start, command=sys.argv, base_commit=BASE, device=device,
              source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()))
    assets_by_fold = {}
    if device == 'cuda':
        configure_cuda()
        torch.cuda.set_per_process_memory_fraction(.40)
    seals = {}
    for fold in FOLDS:
        assets_by_fold[fold] = frozen_assets(refs[TEST, fold], torch.device(device))
        for purpose in PURPOSES:
            ref = refs[purpose, fold]
            seals[f'{purpose}/{fold}'] = _export_fold(ref, *assets_by_fold[fold])
        del assets_by_fold[fold]
        if device == 'cuda':
            torch.cuda.empty_cache()
    immutable(RUN / 'ENDPOINTS_EXPORTED.json', dict(
        folds=seals, config_sha256=digest(RUN / 'RUN_CONFIG.json'), sealed_unix=time.time(),
        truth_reads=0, seconds=time.time()-start, training_updates=conf['training_updates']))
    status(RUN, 'ENDPOINTS_SEALED', windows=sum(seal['windows'] for seal in seals.values()),
           new_ddim_calls=sum(seal['new_ddim_calls'] for seal in seals.values()))


def _verify_all_endpoints(refs):
    export_seal = read(RUN / 'ENDPOINTS_EXPORTED.json')
    require(export_seal['truth_reads'] == 0 and export_seal['training_updates'] == 0
            and export_seal['config_sha256'] == digest(RUN / 'RUN_CONFIG.json'),
            'endpoints were not fully frozen using observations only')
    verified = {}
    for (purpose, fold), ref in refs.items():
        folder = _endpoint_directory(ref)
        path = folder / 'ENDPOINTS_SEALED.json'
        seal = read(path)
        require(digest(path) == export_seal['folds'][f'{purpose}/{fold}']['endpoint_seal_sha256']
                and len(seal['rows']) == seal['windows'] == len(ref.indices) and seal['truth_reads'] == 0,
                'frozen endpoint fold differs')
        for ordinal, (index, row) in enumerate(zip(ref.indices, seal['rows']), 1):
            _validated_endpoints(folder, row, _side_identity(ref, ordinal, index))
        verified[purpose, fold] = seal['rows']
    return export_seal, verified


def _population(refs):
    rows = []
    for fold in FOLDS:
        split = refs[TEST, fold].split
        for purpose in ('train', DEV, TEST):
            counts = Counter(len(split['members'][index]['agent_ids']) for index in split[purpose+'_indices'])
            groups = {group: [n for n in counts if group == 'all' or _group(n) == group] for group in GROUPS}
            groups.update({f'N={n}': [n] for n in sorted(counts)})
            for group, members in groups.items():
                rows.append(dict(fold=fold, split=purpose, group=group, windows=sum(counts[n] for n in members),
                                 agents=sum(n*counts[n] for n in members)))
    return rows


def _read_prior(purpose):
    output = _output(ORIGINAL, purpose)
    rows = {(r['fold'], int(r['seed']), r['window']): r for r in read_csv(output/'PER_WINDOW.csv')}
    groups = {(r['fold'], int(r['seed']), r['group']): r for r in read_csv(output/'PER_SEED.csv')}
    return rows, groups


def _max_diff(actual, prior, fields):
    return max((abs(float(actual[key])-float(prior[key])) for key in fields), default=0.)


def _score_fold(ref, endpoint_rows, prior):
    from .statistics import diagnose_window

    prior_windows, prior_groups = prior
    reader = FormalData(ref.split, phase='scoring', verify=False)
    folder = _endpoint_directory(ref)
    accumulators = {seed: {g: {model: MetricAccumulator() for model in MODELS} for g in GROUPS}
                    for seed in ref.seeds}
    windows, worst_window, worst_group = [], 0., 0.
    for ordinal, (index, scenario) in enumerate(zip(ref.indices, endpoint_rows), 1):
        expected = _side_identity(ref, ordinal, index)
        endpoints = _validated_endpoints(folder, scenario, expected)
        record = ref.split['records'][index]
        require(digest(record['path']) == record['sha256'], 'scoring record changed after export')
        batch = reader.batch([index], ref.purpose, truth=True)
        n = expected['n']
        require(batch.valid.shape == (1,n) and bool(batch.valid.all()), 'scoring mask differs')
        path0, path1 = {}, {}
        for seed in ref.seeds:
            pair, _ = ref.pair(index, seed)
            old = prior_windows[ref.fold, seed, expected['window']]
            require(int(old['n']) == n and old['group'] == _group(n), 'scoring row population differs')
            for model, value in pair.items():
                stats = hard_metric_statistics(torch.from_numpy(value)[None], batch.future, batch.valid)
                values = _row_metrics(stats)
                difference = _max_diff({f'{model}_{k}': v for k,v in values.items()}, old,
                                       (f'{model}_{key}' for key in (*METRICS, 'L','G')))
                require(difference <= NUMERIC_TOL_M, 'per-window archived metric did not replay')
                worst_window = max(worst_window, difference)
                for group in ('all', _group(n)):
                    accumulators[seed][group][model].update_statistics(stats)
            path0[seed] = pair['S0']
            path1[seed] = pair['R12']
        window = diagnose_window(fold=ref.fold, split=ref.purpose, window=expected['window'],
            observed=batch.observed[0].numpy(), future=batch.future[0].numpy(),
            path0=path0, path1=path1, dt=float(ref.split['dt']), **endpoints)
        windows.append(window)
        if ordinal % 100 == 0 or ordinal == len(ref.indices):
            status(RUN, 'SCORING_FROZEN_DIAGNOSIS', fold=ref.fold, split=ref.purpose,
                   windows=ordinal, total=len(ref.indices))
    for seed, by_group in accumulators.items():
        for group, acc in by_group.items():
            value = _group_row(ref.fold, seed, group, acc)
            old = prior_groups[ref.fold, seed, group]
            require(all(value[key] == int(old[key]) for key in ('windows','agents')), 'archived group counts differ')
            if value['windows']:
                keys = [f'{model}_{metric}_m' for model in MODELS for metric in (*METRICS,'L','G')]
                difference = _max_diff(value, old, keys)
                require(difference <= NUMERIC_TOL_M, 'archived group metrics did not replay')
                worst_group = max(worst_group, difference)
    return windows, dict(windows=len(windows), seeds=ref.seeds,
                         max_window_metric_abs_m=worst_window, max_group_metric_abs_m=worst_group)


def _draw_cases(refs, diagnostics):
    from .figures import render_case, select_cases

    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ModuleNotFoundError:
        require_matplotlib_panel_alignment = None
    cases = []
    for fold in FOLDS:
        ref = refs[TEST, fold]
        reader = FormalData(ref.split, phase='scoring', verify=False)
        selected = select_cases([item.row for item in diagnostics[TEST, fold]])
        indices = {ref.rows[index,2035]['window']: index for index in ref.indices}
        for case, row in selected:
            index = indices[row['window']]
            batch = reader.batch([index], TEST, truth=True)
            pair, _ = ref.pair(index, 2035)
            relative = f'cases/{fold}_{case}.png'
            render_case(DOC / relative, fold=fold, case=case, window=row['window'],
                        observed=batch.observed[0].numpy(), truth=batch.future[0].numpy(),
                        path0=pair['S0'], selected_metrics=row, qa_dir=RUN/'case_qa',
                        require_matplotlib_panel_alignment=require_matplotlib_panel_alignment)
            cases.append({key: row[key] for key in ('fold','split','window','n','g0_J_m','g0_L_m','g0_G_m')}
                         | dict(case=case, figure=relative, seed=2035))
    return cases


def _table(path, rows):
    fields = dict.fromkeys(name for row in rows for name in row)
    csvwrite(path, [fields | row for row in rows])


def score(refs):
    from .statistics import summarize_windows

    started = time.time()
    export_seal, verified = _verify_all_endpoints(refs)
    prior = {purpose: _read_prior(purpose) for purpose in PURPOSES}
    first_truth = time.time()
    require(first_truth >= export_seal['sealed_unix'], 'truth preceded endpoint sealing')
    diagnostics = {}
    replays = {}
    for key, ref in refs.items():
        items, verification = _score_fold(ref, verified[key], prior[ref.purpose])
        diagnostics[key] = items
        replays[f'{ref.purpose}/{ref.fold}'] = verification
    DOC.mkdir(parents=True, exist_ok=True)
    items = [item for group in diagnostics.values() for item in group]
    groups = summarize_windows(items)
    _table(DOC/'PER_WINDOW.csv', [item.row for item in items])
    _table(DOC/'GROUPS.csv', groups)
    csvwrite(DOC/'DEMOGRAPHY.csv', _population(refs))
    scored = time.time()
    csvwrite(DOC/'CASES.csv', _draw_cases(refs, diagnostics))
    done = time.time()
    export_folds = export_seal['folds']
    receipt = dict(
        status='COMPLETE', training_updates=0, tested_seeds=5, dev_seeds=1,
        endpoint_seal_sha256=digest(RUN/'ENDPOINTS_EXPORTED.json'),
        original_score_sealed_before_truth=True, export_sealed_unix=export_seal['sealed_unix'],
        first_truth_read_unix=first_truth,
        windows={key: row['windows'] for key,row in replays.items()}, replay_details=replays,
        replay_max_abs_m=max(max(row['max_window_metric_abs_m'],row['max_group_metric_abs_m']) for row in replays.values()),
        new_ddim_calls=sum(row['new_ddim_calls'] for row in export_folds.values()),
        **{key:max(row[key] for row in export_folds.values()) for key in
           ('endpoint_replay_max_abs_m','cross_seed_endpoint_max_abs_m','decomposition_max_abs_m')},
        seconds=dict(export=export_seal['seconds'], scoring=scored-started,cases=done-scored,
                     total=done-read(RUN/'START.json')['unix']),
        per_fold_export=export_folds,
        output_sha256={f.name:digest(f) for f in sorted(DOC.glob('*.csv'))},
    )
    write(DOC/'RECEIPT.json',receipt)
    write(DOC/'RUN_CONFIG.json',read(RUN/'RUN_CONFIG.json'))
    status(RUN, 'DIAGNOSIS_SCORED', windows=len(items), new_ddim_calls=receipt['new_ddim_calls'])


def publish():
    from .report import build_report

    receipt = read(DOC/'RECEIPT.json')
    require(receipt['status'] == 'COMPLETE' and len(list((DOC/'cases').glob('*.png'))) == 6,
            'case figures or scores are incomplete')
    require(all(digest(DOC/name) == sha for name, sha in receipt['output_sha256'].items()),
            'diagnosis table changed')
    build_report(DOC)
    write(DOC/'ARTIFACT_SHA256.json', {str(p.relative_to(DOC)):digest(p)
          for p in sorted(DOC.rglob('*')) if p.is_file() and p.name != 'ARTIFACT_SHA256.json'})
    require(subprocess.check_output(['git','branch','--show-current'],text=True).strip() == BRANCH,
            'wrong diagnosis archive branch')
    subprocess.run(['git','add',str(DOC.relative_to(SOURCE))],check=True)
    subprocess.run(['git','diff','--cached','--check'],check=True)
    if subprocess.call(['git','diff','--cached','--quiet']):
        subprocess.run(['git','commit','-m','Archive frozen HOTEL and UNIV prediction diagnosis'],check=True)
    subprocess.run(['git','push','-u','origin',BRANCH],check=True)
    sha = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    remote = subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],text=True).split()[0]
    require(sha == remote, 'diagnosis remote SHA differs')
    write(RUN/'DELIVERY.json',dict(sha=sha,verified_remote_sha=remote,branch=BRANCH))
    status(RUN,'COMPLETE',sha=sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('export','score','publish'))
    parser.add_argument('--device', choices=('cuda','cpu'), default='cuda')
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    with (RUN/'diagnosis.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase == 'publish':
            publish()
        elif args.phase == 'export':
            with (ASSETS/'auxiliary_gpu.lock').open('a') as resource:
                if args.device == 'cuda':
                    fcntl.flock(resource,fcntl.LOCK_EX)
                export(_make_references(),args.device)
        else:
            score(_make_references())


if __name__ == '__main__':
    main()
