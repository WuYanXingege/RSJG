"""Replay HOTEL epoch eight, then continue the identical fit only if warranted."""
import argparse
from copy import deepcopy
import fcntl
from pathlib import Path
import subprocess
import sys
import time
import traceback

from bcep.contracts import digest
from bcep_joint.common import read, write
from joint_flow_fourfold.common import immutable, status
from k20_dual_finetune.run import (
    ASSETS, BRANCH, HISTORICAL, PRIOR, SOURCE, gpu_free,
    main_waiting_for_gpu, protect as protect_prior_run,
)

BASE = 'c0391f8db015ce7583a54e8a85b518fb6358d76b'
PARENT_SHA = '863d169a44122fbe40668d2299202b39d605dbae708a62f2da8e2c20eb6797cc'
PARENT = ASSETS / 'k20_dual_finetune_20261125'
RUN = ASSETS / 'hotel_k20_followup_20261125'
DOC = SOURCE / 'docs/research/continuous_joint_flow/k20_dual_finetune_hotel_followup_20261125'
ORIGINAL_DOC = SOURCE / 'docs/research/continuous_joint_flow/k20_dual_finetune_20261125'
SCOPE = '看到 HOTEL 第 8 轮终点双降后追加的单种子探索性开发评估'
METRICS = ('JADE', 'JFDE', 'minADE', 'minFDE')
YIELD_TO_MAIN = 75


def sources():
    return {str(path.relative_to(SOURCE)): digest(path)
            for path in sorted((SOURCE / 'hotel_k20_followup').glob('*.py'))}


def protect(config):
    protect_prior_run(read(PARENT / 'RUN_CONFIG.json'))
    assert sources() == config['source_sha256']
    assert digest(config['parent_checkpoint']) == PARENT_SHA
    e = PARENT / 'hotel/evaluation'
    assert digest(e / 'ENDPOINT_GATE.json') == config['original_endpoint_gate_sha256']
    assert digest(e / 'endpoints/SCORE_RECEIPT.json') == config['original_score_receipt_sha256']
    assert digest(e / 'endpoints/PREDICTIONS_FROZEN.json') == config['original_endpoint_seal_sha256']
    artifact_manifest = ORIGINAL_DOC / 'ARTIFACT_SHA256.json'
    assert digest(artifact_manifest) == config['original_archive_manifest_sha256']
    for path, sha in read(artifact_manifest).items():
        assert digest(ORIGINAL_DOC / path) == sha


def prepare():
    from k20_dual_finetune.train import RECIPE

    prior_config = read(PARENT / 'RUN_CONFIG.json')
    original = PARENT / 'hotel'
    training = read(original / 'TRAINING_RESULT.json')
    evaluation = original / 'evaluation'
    gate = read(evaluation / 'ENDPOINT_GATE.json')
    receipt = read(evaluation / 'endpoints/SCORE_RECEIPT.json')
    assert training['final_checkpoint_sha256'] == PARENT_SHA
    assert training['completed_epochs'] == 8 and training['updates'] == 1560
    assert training['recipe'] == RECIPE
    assert gate['endpoint_gate_passed'] is False and gate['pass_JFDE'] is True
    assert receipt['status'] == 'COMPLETE' and receipt['files'] == 445 and receipt['seed'] == 2035
    assert receipt['manifest_sha256'] == digest(evaluation / 'endpoints/PREDICTIONS_FROZEN.json')
    assert gate['endpoint_score_receipt_sha256'] == digest(evaluation / 'endpoints/SCORE_RECEIPT.json')
    immutable(RUN / 'RUN_CONFIG.json', dict(
        base_commit=BASE, reporting_scope=SCOPE, fold='hotel', seed=2035, M=256, K=20,
        windows=445, agents=1197, single_windows=144, multi_windows=301,
        parent_checkpoint=training['final_checkpoint'], parent_sha256=PARENT_SHA,
        parent_epoch=8, parent_updates=1560, target_epoch_on_pass=16,
        original_archive_manifest_sha256=digest(ORIGINAL_DOC / 'ARTIFACT_SHA256.json'),
        original_endpoint_gate_sha256=digest(evaluation / 'ENDPOINT_GATE.json'),
        original_score_receipt_sha256=digest(evaluation / 'endpoints/SCORE_RECEIPT.json'),
        original_endpoint_seal_sha256=receipt['manifest_sha256'],
        original_historical_csv_sha256=prior_config['identities']['hotel']['historic_per_seed_sha256'],
        original_training_recipe=RECIPE, teacher_identity=prior_config['identities']['hotel'],
        source_sha256=sources(), environment=prior_config['environment'],
        continuation_gate='epoch8 JADE, JFDE, minADE, minFDE each strictly lower than same-seed B',
        extension_value='epoch16 all four <= epoch8, at least one strictly lower',
        historical_target_minFDE_reduction_m=.02, ddim_steps=20, eta=0.,
        endpoint_replay_tolerance=dict(atol=1e-6, rtol=1e-6),
        cuda_allocator_fraction=prior_config['cuda_allocator_fraction'],
        min_free_gpu_mib=prior_config['min_free_gpu_mib'],
    ))
    config = read(RUN / 'RUN_CONFIG.json')
    protect(config)
    return config


def student_from_checkpoint(provider, teacher, path, sha, epoch):
    import torch

    assert digest(path) == sha
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert state['completed_epoch'] == epoch
    assert state['update'] == 195 * epoch
    assert state['cursor'] == 0 and state['order'] is None
    student_provider = deepcopy(provider)
    student = deepcopy(teacher)
    student_provider.head.readout[-1].load_state_dict(state['readout'], strict=True)
    student.load_state_dict(state['dependence'], strict=True)
    return student_provider, student


def worker():
    import torch
    from bcep_v2.runtime import configure_cuda
    from joint_flow_dependency.model import EdgeDependence
    from joint_flow_fourfold.inputs import Inputs
    from .continuation import continue_fold
    from .evaluation import full_evaluation

    config = read(RUN / 'RUN_CONFIG.json')
    protect(config)
    torch.set_num_threads(2)
    configure_cuda()
    torch.cuda.set_per_process_memory_fraction(config['cuda_allocator_fraction'])
    identity = config['teacher_identity']
    manifest = PRIOR / 'hotel/UPSTREAM_FROZEN.json'
    assert digest(manifest) == identity['upstream_sha256']
    assert digest(identity['checkpoint']) == identity['teacher_sha256']
    inputs = Inputs(manifest, RUN, 'cuda')
    model, provider = inputs.load_frozen('cuda')
    teacher = EdgeDependence()
    saved = torch.load(identity['checkpoint'], map_location='cpu', weights_only=False)
    assert saved['completed_epoch'] == 19
    teacher.load_state_dict(saved['model'], strict=True)
    teacher.to('cuda').eval().requires_grad_(False)
    status(RUN, 'EPOCH8_FULL_EVALUATION')
    student_provider, student = student_from_checkpoint(
        provider, teacher, config['parent_checkpoint'], PARENT_SHA, 8)
    eight = full_evaluation(inputs, model, student_provider, student, RUN / 'epoch08',
                           PARENT_SHA, 8, HISTORICAL['hotel'],
                           old_evaluation=PARENT / 'hotel/evaluation', device='cuda')
    del student_provider, student
    delta = eight['groups']['all']['delta']
    passed = all(delta[name] < 0 for name in METRICS)
    assert passed == eight['all_four_improved']
    immutable(RUN / 'CONTINUATION_GATE.json', dict(
        passed=passed, epoch=8, delta_m={name: delta[name] for name in METRICS},
        minFDE_2cm_goal_met=delta['minFDE'] <= -.02,
        evaluation_sha256=digest(RUN / 'epoch08/SUMMARY.json'), parent_sha256=PARENT_SHA,
    ))
    if passed:
        status(RUN, 'CONTINUATION_TRAINING', parent_epoch=8, target_epoch=16)
        training = continue_fold(inputs, model, provider, teacher, PARENT / 'hotel',
                                 RUN / 'continuation', PARENT_SHA,
                                 device='cuda', should_yield=main_waiting_for_gpu)
        if training is None or main_waiting_for_gpu():
            return YIELD_TO_MAIN
        assert training['completed_epochs'] == 16 and training['updates'] == 3120
        assert training['new_updates'] == 1560
        sha = training['final_checkpoint_sha256']
        student_provider, student = student_from_checkpoint(
            provider, teacher, training['final_checkpoint'], sha, 16)
        status(RUN, 'EPOCH16_FULL_EVALUATION')
        sixteen = full_evaluation(inputs, model, student_provider, student, RUN / 'epoch16',
                                  sha, 16, HISTORICAL['hotel'], device='cuda')
        all8, all16 = eight['groups']['all']['new'], sixteen['groups']['all']['new']
        changes = {name: all16[name] - all8[name] for name in METRICS}
        better = all(v <= 0 for v in changes.values()) and any(v < 0 for v in changes.values())
        decision = dict(continued=True, additional_updates=training['new_updates'],
            final_epoch=16, delta_16_minus_8_m=changes, extension_further_improved=better,
            full_metrics_16_lower_than_B=sixteen['all_four_improved'],
            score_16_sha256=digest(RUN / 'epoch16/SUMMARY.json'),
            ddim_calls=eight['ddim_calls'] + sixteen['ddim_calls'])
    else:
        decision = dict(continued=False, additional_updates=0, final_epoch=8,
                        blocking_metrics=[name for name in METRICS if delta[name] >= 0],
                        ddim_calls=eight['ddim_calls'])
    protect(config)
    immutable(RUN / 'RESULT.json', dict(**decision,
        epoch8_evaluation_updates=0, epoch8_score_sha256=digest(RUN / 'epoch08/SUMMARY.json')))
    return 0


def archive():
    from .report import build_report
    build_report()
    assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == BRANCH
    assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True).strip()
    subprocess.run(['git', 'add', str(DOC.relative_to(SOURCE))], check=True)
    subprocess.run(['git', 'diff', '--cached', '--check'], check=True)
    if subprocess.call(['git', 'diff', '--cached', '--quiet']):
        subprocess.run(['git', 'commit', '-m', 'Archive HOTEL full trajectory follow-up'], check=True)
    subprocess.run(['git', 'push', '-u', 'origin', BRANCH], check=True)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/' + BRANCH],
                                     text=True).split()[0]
    assert remote == sha
    write(RUN / 'DELIVERY.json', dict(sha=sha, verified_remote_sha=remote, branch=BRANCH))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        return worker()
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'controller.lock').open('a') as controller:
        fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = prepare()
        if not (RUN / 'START.json').exists():
            write(RUN / 'START.json', dict(unix=time.time(),
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                command=[sys.executable, '-m', 'hotel_k20_followup.run']))
        attempts = 0
        while not (RUN / 'RESULT.json').exists():
            while main_waiting_for_gpu() or gpu_free() < config['min_free_gpu_mib']:
                status(RUN, 'WAITING_FOR_GPU')
                time.sleep(30)
            with (ASSETS / 'auxiliary_gpu.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if main_waiting_for_gpu() or gpu_free() < config['min_free_gpu_mib']:
                    continue
                cmd = [sys.executable, '-m', 'hotel_k20_followup.run', '--worker']
                with (RUN / 'worker.log').open('a', buffering=1) as log:
                    process = subprocess.Popen(cmd, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT)
                    write(RUN / 'ACTIVE_CHILD.json', dict(pid=process.pid, command=cmd, unix=time.time()))
                    code = process.wait()
            if code == YIELD_TO_MAIN:
                status(RUN, 'YIELDED_TO_MAIN_STAGE')
            elif code:
                attempts += 1
                if attempts >= 3:
                    raise RuntimeError('HOTEL follow-up failed three attempts; see worker.log')
                status(RUN, 'RESUMING_SAME_RUN', attempt=attempts + 1, last_exit_code=code)
            else:
                assert (RUN / 'RESULT.json').exists()
                break
            time.sleep(30)
        protect(config)
        write(RUN / 'WALL_TIME.json', dict(seconds=time.time() - read(RUN / 'START.json')['unix']))
        status(RUN, 'ARCHIVING')
        archive()
        status(RUN, 'COMPLETE', **read(RUN / 'RESULT.json'))
    return 0


if __name__ == '__main__':
    try:
        code = main()
    except Exception:
        if '--worker' not in sys.argv:
            status(RUN, 'INTERRUPTED_RESUME_SAME_RUN', traceback=traceback.format_exc())
        raise
    raise SystemExit(code)
