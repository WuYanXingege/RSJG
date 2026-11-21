"""Prepare, run exactly the bounded audit, compare, and produce receipts."""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from .common import BASE, SOURCE, OUT, PUBLIC, FROZEN, FROZEN_SHA, read, write, sha, frozen, exact
from .fixtures import prepare as prepare_synthetic

PROTECTED = ['bcep_ebjd_v2_integrated', 'zara2_bcep_v3_stage2_20261120', 'zara2_costproxy_stage2_20261119',
    'zara2_sharedworld_stage2_onerun_20261119', 'zara2_v3_frozen_feedback_audit_20261121',
    'joint_score_identifiability_20261121', 'ssec_network_acceptance_20261121', 'ssec_candidate_identity_20261121',
    'zara2_ssec_energy_stage2_20261121', 'zara2_ssec_energy_stage2_readiness_20261121']


def protection():
    root = SOURCE.parent / 'RSJG_EBJD_assets/experiments'
    manifest = {}
    for name in PROTECTED:
        assert (root / name).is_dir(), name
        for p in sorted((root / name).rglob('*')):
            if p.is_file():
                manifest[str(p.relative_to(root))] = dict(bytes=p.stat().st_size, sha256=sha(p))
    return manifest


def journal(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def committed_journal(rows, committed_update):
    """Read-only counterpart of native last-attempt-wins selection.

    Native epoch_summary correctly rejects a four-update incomplete epoch.
    Here the committed checkpoint supplies the horizon; we never fabricate
    another 164 records to call it a complete epoch.
    """
    selected = {}
    for r in rows:
        if 1 <= r['update'] <= committed_update:
            selected[r['update']] = r
    assert set(selected) == set(range(1, committed_update + 1)), 'missing committed update'
    return [selected[i] for i in range(1, committed_update + 1)]


def semantic_journal(r):
    # Explicitly variable operational telemetry is NOT model/train state.
    # Retain numeric gradient, parameter-change and scene energy statistics.
    return dict(update=r['update'], epoch=r['epoch'], modules=r['modules'], physical_groups=[
        dict(valid_people=g['valid_people'], scenes=g['scenes']) for g in r['physical_groups']])


def csvwrite(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)


def prepare():
    assert not OUT.exists(), 'audit exists; use its existing checkpoints, do not start another experiment'
    # Also allow the published audit commit descended from BASE; all runtime
    # implementation hashes below must still equal the fixed original ones.
    subprocess.run(['git', 'merge-base', '--is-ancestor', BASE, 'HEAD'], cwd=SOURCE, check=True)
    cfg = frozen()
    OUT.mkdir(parents=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    p = protection()
    write(OUT / 'PROTECTED_BEFORE.json', p)
    manifest = prepare_synthetic()
    write(PUBLIC / 'SYNTHETIC_INPUTS.json', dict(synthetic_only=True, real_optimizer_inputs=0,
        synthetic_epoch_size=2681, materialized_samples=64, logical_batch_size=16,
        each_logical_batch_agent_counts=manifest['logical_pattern'], maps=[6,256,256], K=20,
        dataset_manifest_sha256=sha(OUT / 'synthetic_inputs/MANIFEST.json'),
        input_rows=manifest['materialized_rows'], construction=manifest['construction_seed_namespace']))
    write(PUBLIC / 'AUDIT_CONFIG.json', dict(base=BASE, official_frozen_config_sha256=FROZEN_SHA,
        unchanged_model_config=cfg['config'], unchanged_recipe=cfg['recipe'], initialization_sha256=cfg['initial_model_sha256'],
        architecture='SSEC energy_only', rejection_gradient='proposal_st', source_manifest=cfg['implementation_files'],
        authorized_scope='synthetic-only bounded recovery, no real-data training',
        phases=dict(continuous=4,partial=2,resume=2,replay_crash=1,replay_resume=2),
        total_authorized_physical_optimizer_steps=11, logical_updates_per_comparison=4,
        formal_authorization_flag_set=False, record_progress=False,
        record_progress_reason='wall-clock progress only; complete numeric checkpoint compared with zero skipped keys'))
    write(OUT / 'PREPARED.json', dict(passed=True, protected_files=len(p), seconds=time.perf_counter()-started))
    print('Prepared independent inputs, protected assets, unchanged source/config; optimizer updates 0', flush=True)


def run_phases():
    assert (OUT / 'PREPARED.json').exists()
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(SOURCE / 'scripts'))
    assert env.get('RUN_ONE_ZARA2_SSEC') != 'true', 'formal authorization must remain unset'
    for phase, arm in [('continuous','continuous'),('partial','split'),('resume','split'),
                       ('replay_crash','replay'),('replay_resume','replay')]:
        if (OUT / arm / (phase + '_RECEIPT.json')).exists():
            continue
        if phase == 'resume':
            assert not (OUT / 'split/checkpoint_update_002.pt').exists()
            shutil.copy2(OUT / 'split/last.pt', OUT / 'split/checkpoint_update_002.pt')
            shutil.copy2(OUT / 'split/MODULE_UPDATES.jsonl', OUT / 'split/MODULE_UPDATES_at_002.jsonl')
        if phase == 'replay_crash':
            (OUT / 'replay').mkdir(exist_ok=False)
            shutil.copy2(OUT / 'split/checkpoint_update_002.pt', OUT / 'replay/last.pt')
            shutil.copy2(OUT / 'split/MODULE_UPDATES_at_002.jsonl', OUT / 'replay/MODULE_UPDATES.jsonl')
        result = subprocess.run([sys.executable, '-m', 'ssec_nonzero.worker', phase], cwd=SOURCE, env=env)
        if result.returncode:
            write(PUBLIC / 'READINESS.json', dict(status='BLOCKED', phase=phase,
                reason='synthetic worker failed; inspect the exact failure before any additional updates', real_optimizer_updates=0))
            raise SystemExit(result.returncode)


def compare():
    def load(path):
        return torch.load(path, map_location='cpu', weights_only=False)

    states = {arm: load(OUT / arm / 'last.pt') for arm in ['continuous','split','replay']}
    full_rows = []
    for arm in ['split','replay']:
        # No exclusion list: compare the ENTIRE final native checkpoint.
        exact(states['continuous'], states[arm], f'continuous_vs_{arm}')
        for key in states['continuous']:
            full_rows.append(dict(comparison=f'continuous_vs_{arm}', field=key, exact=True, tolerance=0))
    assert states['split']['update'] == 4 and states['split']['cursor'] == 64
    two = load(OUT / 'split/checkpoint_update_002.pt')
    assert two['update'] == 2 and two['cursor'] == 32
    assert states['split']['epoch_steps'] == 4 and states['split']['epoch_counts']['valid_scene_count'] == 64
    assert states['split']['scheduler']['factor'] == .2 and states['split']['scheduler']['epoch'] == 1
    # Per-step model + buffer + AdamW + all RNG + loss/gradient/decision data.
    comparisons = []
    for update in range(1,5):
        a = load(OUT / 'continuous' / f'continuous_update_{update:03d}.pt')
        phase = 'partial' if update <= 2 else 'resume'
        b = load(OUT / 'split' / f'{phase}_update_{update:03d}.pt')
        exact(a, b, f'step_{update}')
        comparisons.append(dict(update=update, model_buffers_optimizer_rng_statistics_exact=True,
            clipped_gradients_exact=True, candidate_scores_and_decisions_exact=True,
            parameter_tensors=len(a['parameter_names']), buffer_tensors=len(a['buffer_names'])))
    a3 = load(OUT / 'continuous/continuous_update_003.pt')
    for phase in ['replay_crash','replay_resume']:
        exact(a3, load(OUT / 'replay' / f'{phase}_update_003.pt'), phase)
    exact(load(OUT / 'continuous/continuous_update_004.pt'), load(OUT / 'replay/replay_resume_update_004.pt'), 'replayed_step4')
    # Native checkpoint contains genuine nonempty, nonzero AdamW moments.
    opt = states['split']['optimizer']
    assert opt['state'] and all(float(x['step']) == 4 for x in opt['state'].values())
    nonzero_moments = sum(bool(x['exp_avg'].any()) and bool(x['exp_avg_sq'].any()) for x in opt['state'].values())
    assert nonzero_moments == len(opt['state'])
    continuous_rows = journal(OUT / 'continuous/MODULE_UPDATES.jsonl')
    split_rows = journal(OUT / 'split/MODULE_UPDATES.jsonl')
    replay_rows = journal(OUT / 'replay/MODULE_UPDATES.jsonl')
    assert [r['update'] for r in continuous_rows] == [1,2,3,4]
    assert [r['update'] for r in split_rows] == [1,2,3,4]
    assert [r['update'] for r in replay_rows] == [1,2,3,3,4]
    for rr in [split_rows, committed_journal(replay_rows, 4)]:
        exact([semantic_journal(r) for r in continuous_rows], [semantic_journal(r) for r in rr], 'MODULE_UPDATES_numeric')
    # At the intentional crash, future/uncommitted update3 must not be counted.
    assert [r['update'] for r in committed_journal(replay_rows[:3], 2)] == [1,2]
    assert replay_rows[2]['session'] != replay_rows[3]['session']
    from bcep_scene_energy.formal_monitor import StepObserver
    observer = StepObserver(torch.nn.Module(), OUT / 'replay/MODULE_UPDATES.jsonl', 4)
    try:
        observer.epoch_summary(1)
    except RuntimeError as e:
        assert 'incomplete' in str(e)
    else:
        raise AssertionError('native monitor counted four updates as a full epoch')
    identity = read(OUT / 'continuous/DDIM_IDENTITY.json')
    exact(identity, read(OUT / 'split/DDIM_IDENTITY.json'), 'DDIM_split')
    exact(identity, read(OUT / 'replay/DDIM_IDENTITY.json'), 'DDIM_replay')
    assert [r['N'] for r in identity] == [1,2,9,57]
    phases = [('continuous','continuous'),('partial','split'),('resume','split'),('replay_crash','replay'),('replay_resume','replay')]
    receipts = [read(OUT / arm / (phase + '_RECEIPT.json')) for phase,arm in phases]
    assert all(r['passed'] and r['real_optimizer_updates'] == 0 for r in receipts)
    events = journal(OUT / 'OPTIMIZER_EVENTS.jsonl')
    assert len(events) == sum(r['actual_synthetic_optimizer_updates'] for r in receipts) == 11
    shutil.copy2(OUT / 'OPTIMIZER_EVENTS.jsonl', PUBLIC / 'OPTIMIZER_EVENTS.jsonl')
    gradients = []
    for phase,arm in phases:
        for r in journal(OUT / arm / 'MODULE_UPDATES.jsonl'):
            # Full journal exported separately; gradient table is unique by
            # arm/session/update, so copied prefix events are not extra steps.
            if phase not in ['continuous','partial','replay_crash']:
                continue
            for module, m in r['modules'].items():
                assert all(np.isfinite(v) for v in m.values())
                assert m['gradient_l2'] > 0 and m['parameter_delta_l2'] > 0
                gradients.append(dict(arm=arm,session=r['session'],update=r['update'],module=module,**m))
    csvwrite(PUBLIC / 'FULL_STATE_COMPARISON.csv', full_rows)
    csvwrite(PUBLIC / 'MODULE_GRADIENTS_AND_UPDATES.csv', gradients)
    write(PUBLIC / 'PER_UPDATE_EXACT.json', comparisons)
    write(PUBLIC / 'DDIM_IDENTITY.json', identity)
    write(PUBLIC / 'PHASE_RECEIPTS.json', receipts)
    for arm in ['continuous','split','replay']:
        shutil.copy2(OUT / arm / 'MODULE_UPDATES.jsonl', PUBLIC / f'MODULE_UPDATES_{arm}.jsonl')
    write(PUBLIC / 'JOURNAL_REPLAY.json', dict(passed=True, continuous=[1,2,3,4],split=[1,2,3,4],replay=[1,2,3,3,4],
        committed_at_crash=[1,2], committed_after_replay=[1,2,3,4],
        both_update3_attempts_preserved=True, latest_session_selected=True, all_numeric_telemetry_exact=True,
        native_incomplete_epoch_rejected=True, no_fabricated_epoch_records=True,
        variable_operational_fields=['session','unix','logical_group_forward_backward_seconds','allocated_peak_bytes','coupling_forward_seconds']))
    before = read(OUT / 'PROTECTED_BEFORE.json')
    after = protection()
    exact(before, after, 'protected_assets')
    write(OUT / 'PROTECTED_AFTER.json', after)
    protected = dict(unchanged=True, files=len(after), bytes=sum(r['bytes'] for r in after.values()),
        manifest_sha256=sha(OUT / 'PROTECTED_AFTER.json'), roots=PROTECTED)
    write(PUBLIC / 'ASSET_PROTECTION.json', protected)
    frozen()
    assert not subprocess.check_output(['git','diff',BASE,'--','bcep_scene_energy','bcep_v2','bcep_v3','bcep_joint','ebjd',str(FROZEN.relative_to(SOURCE))],cwd=SOURCE)
    readiness = dict(status='PASS_SYNTHETIC_NONZERO_RECOVERY', base=BASE,
        real_optimizer_updates=0, actual_synthetic_optimizer_updates=11, completed_epochs=0,
        independent_training_seed_count=1, seed=3201, steps_continuous=4, steps_split=[2,2], replay_additional_physical_steps=3,
        full_native_checkpoint_exact=True, compared_native_fields=list(states['continuous']), omitted_native_fields=[],
        per_step_model_buffer_optimizer_rng_statistics_exact=True, parameter_count=receipts[0]['parameters'],
        model_buffers=len(receipts[0]['buffers']), nonzero_AdamW_state_entries=nonzero_moments,
        every_step_phi_rho_encoder_gradient_and_update_finite_nonzero=True,
        fixed_candidates_slots_keyed_noise_and_energy_checks=True, journal_replay_passed=True,
        frozen_formal_config_sha256=FROZEN_SHA, architecture_objective_optimizer_controller_unchanged=True,
        protected_assets_unchanged=True, formal_training_started=False, formal_training_authorized=False,
        record_progress=False, checkpoint_comparison_tolerance=0,
        limitations=['four logical updates in epoch1 only; no completed epoch or new model selection',
                    'native wall-clock progress fields disabled for both arms; real journal timing varies and is preserved separately',
                    'same GPU/software deterministic FP32; not a cross-device bitwise guarantee',
                    'non-atomic partial JSONL writes or disk loss were not injected; complete-record replay was tested'],
        unresolved_blockers=[])
    write(PUBLIC / 'READINESS.json', readiness)
    write(OUT / 'READINESS.json', readiness)
    print('PASS_SYNTHETIC_NONZERO_RECOVERY: full checkpoint exact; 11 synthetic / 0 real updates', flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['prepare','run','compare']); a = p.parse_args()
    try:
        {'prepare':prepare,'run':run_phases,'compare':compare}[a.action]()
    except Exception as e:
        write(PUBLIC / 'READINESS.json', dict(status='BLOCKED',action=a.action,reason=repr(e),real_optimizer_updates=0))
        raise


if __name__ == '__main__':
    main()
