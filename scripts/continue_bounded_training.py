#!/usr/bin/env python3
"""CPU-only gate, then natural-boundary handoff to the unchanged supervisor.

Never modifies training source/HEAD/config/checkpoints. No optimizer updates here.
The original supervisor must release its lock after its complete epoch5 save.
"""
import argparse
import contextlib
import copy
import csv
from datetime import datetime
from zoneinfo import ZoneInfo
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

METRICS=('minADE','minFDE','JADE','JFDE')


def atomic(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(obj,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def decide(epoch1,epoch2,epoch5):
    for a in (epoch1,epoch2,epoch5):
        if a['seed']!=2035 or a['scenes']<=0 or a['agent_occurrences']<=0:
            raise ValueError('invalid validation identity')
        if not a['scene_ids'] or len(a['scene_ids'])!=a['scenes']:
            raise ValueError('missing scene identities')
        if any(not math.isfinite(a['H'][k]) or a['H'][k]<0 for k in METRICS):
            raise ValueError('invalid validation metrics')
    for a in (epoch2,epoch5):
        if any(a[k]!=epoch1[k] for k in ('seed','scenes','agent_occurrences','scene_ids')):
            raise ValueError('validation membership/order differs')
    reference={k:(epoch1['H'][k]+epoch2['H'][k])/2 for k in METRICS}
    current={k:epoch5['H'][k] for k in METRICS}
    passed={k:current[k]<reference[k] for k in METRICS}
    return dict(pass_gate=all(passed.values()),method='H',comparison='strictly_lower_than_mean_of_epochs_1_and_2',
        reference_mean=reference,epoch5=current,delta={k:current[k]-reference[k] for k in METRICS},
        metric_pass=passed,seed=2035,unit='metre',scope='engineering continuation decision, not method acceptance')


def continuation_manifest(original,gate_path):
    m=copy.deepcopy(original)
    if m.get('bounded_epochs')!=5:raise ValueError('expected the authorized 5-epoch launch')
    cmd=m['command'];index=cmd.index('--stop-after-epoch')
    if cmd[index+1]!='5':raise ValueError('unexpected original training bound')
    cmd[index+1]='100';m.pop('bounded_epochs')
    if '--resume' in cmd:raise ValueError('supervisor adds resume; do not duplicate flag')
    m.update(authorized_total_epochs=100,old_model_control=False,
        continuation_gate=str(gate_path),continuation_gate_sha256=sha(gate_path),
        continuation_authorization='User: four H validation metrics improve over mean of first two epochs',
        formal_inference_five_seed_evaluation_scheduled=True)
    return m


@contextlib.contextmanager
def lock(path,blocking=False):
    with Path(path).open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield f


def check_source(manifest,policy):
    paper=Path(manifest['release_directory'])
    head=subprocess.check_output(['git','-C',str(paper),'rev-parse','HEAD'],text=True).strip()
    if head!=policy['source_commit'] or head!=manifest['workspace_head']:
        raise RuntimeError('source HEAD changed; refusing automatic continuation')
    subprocess.run(['git','-C',str(paper),'diff','--exit-code','HEAD','--','ebjd','scripts','configs','network_release'],check=True)
    for path,digest in manifest['launch_file_hashes'].items():
        if sha(path)!=digest:raise RuntimeError('bound file changed: '+path)


def self_test():
    import unittest
    class Tests(unittest.TestCase):
        def fixture(self,x):
            return dict(seed=2035,scenes=1,agent_occurrences=2,scene_ids=['s1'],H=dict.fromkeys(METRICS,x))
        def test_pass_is_all_four(self):
            d=decide(self.fixture(2),self.fixture(4),self.fixture(2.9));self.assertTrue(d['pass_gate'])
            a=self.fixture(2.9);a['H']['JFDE']=3
            self.assertFalse(decide(self.fixture(2),self.fixture(4),a)['pass_gate'])
        def test_no_absolute_threshold(self):
            self.assertTrue(decide(self.fixture(100),self.fixture(200),self.fixture(149))['pass_gate'])
        def test_invalid_data(self):
            a=self.fixture(1);a['H']['minADE']=float('nan')
            with self.assertRaises(ValueError):decide(self.fixture(2),self.fixture(2),a)
            a=self.fixture(1);a['scene_ids']=['other']
            with self.assertRaises(ValueError):decide(self.fixture(2),self.fixture(2),a)
        def test_launch_only_changes_bound(self):
            with tempfile.TemporaryDirectory() as temp:
                p=Path(temp)/'gate.json';atomic(p,{'pass_gate':True})
                original=dict(bounded_epochs=5,command=['python','train.py','--stop-after-epoch','5'],workspace_head='abc')
                new=continuation_manifest(original,p)
                self.assertEqual(original['command'][-1],'5');self.assertEqual(new['command'][-1],'100')
                self.assertNotIn('bounded_epochs',new);self.assertEqual(new['workspace_head'],'abc')
        def test_lock_prevents_duplicate(self):
            with tempfile.TemporaryDirectory() as temp:
                p=Path(temp)/'lock'
                with lock(p):
                    with self.assertRaises(BlockingIOError):
                        with lock(p):pass
        def test_full_handoff_and_gate_failure_without_launching_training(self):
            import torch
            from unittest.mock import patch
            for success in (True,False):
                with tempfile.TemporaryDirectory() as temp:
                    root=Path(temp);folder=root/'continuation';folder.mkdir()
                    train=root/'training';(train/'validation').mkdir(parents=True)
                    (root/'logs').mkdir();(root/'identity').mkdir()
                    identity=dict(source_commit='test',epochs=100,rollout=False,projection=False)
                    atomic(train/'RUN_IDENTITY.json',identity)
                    checkpoint=train/'last.pt'
                    torch.save(dict(epoch=5,update_index=5,run_identity=identity,
                        scheduler_state=dict(total_epochs=100,completed_epoch=5),
                        optimizer=dict(state={0:{'test':1}}),rng={'test':1}),checkpoint)
                    frozen=train/'fixed_epoch005.pt';frozen.write_bytes(checkpoint.read_bytes())
                    atomic(root/'BOUNDED_RESULT.json',dict(status='BOUNDED_TRAINING_COMPLETE',completed_epochs=5,
                        checkpoint=str(frozen),checkpoint_sha256=sha(checkpoint),updates=5))
                    for epoch,value in ((1,2.),(2,4.),(5,2. if success else 4.)):
                        atomic(train/f'validation/epoch_{epoch:03d}.json',self.fixture(value))
                    original=dict(bounded_epochs=5,command=[sys.executable,'train.py','--stop-after-epoch','5'],
                        workspace_head='test',release_directory=str(root),training_output=str(train),launch_file_hashes={})
                    atomic(root/'RUN_MANIFEST.json',original)
                    (folder/'ORIGINAL_RUN_MANIFEST.json').write_bytes((root/'RUN_MANIFEST.json').read_bytes())
                    atomic(root/'identity/RELEASE_IDENTITY.json',{'epochs_to_execute':5})
                    atomic(folder/'POLICY.json',dict(run_root=str(root),source_commit='test',controller_sha256=sha(__file__),
                        original_manifest_sha256=sha(root/'RUN_MANIFEST.json')))
                    class FakeProcess:
                        pid=424242
                        def __init__(self,*args,**kwargs):
                            atomic(root/'PIPELINE_STATE.json',dict(supervisor_pid=self.pid,status='TRAINING_OR_EVALUATING'))
                        def poll(self):return None
                    with patch(__name__+'.check_source'),patch('subprocess.Popen',side_effect=FakeProcess) as launch:
                        main(root)
                        if success:
                            self.assertEqual(launch.call_count,1)
                            self.assertEqual(read(folder/'STATUS.json')['status'],'HANDED_OFF')
                            self.assertEqual(read(root/'RUN_MANIFEST.json')['command'][-1],'100')
                            self.assertEqual(read(folder/'CONTINUATION_RELEASE_IDENTITY.json')['authorized_remaining_epochs'],95)
                            main(root);self.assertEqual(launch.call_count,1)
                        else:
                            launch.assert_not_called()
                            self.assertEqual(read(folder/'STATUS.json')['status'],'GATE_NOT_PASSED')
                            self.assertEqual(read(root/'RUN_MANIFEST.json'),original)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():raise SystemExit(1)


def main(root):
    root=Path(root).resolve();folder=root/'continuation';policy=read(folder/'POLICY.json')
    if str(root)!=policy['run_root']:raise ValueError('run identity mismatch')
    state_path=folder/'STATUS.json';manifest_path=root/'RUN_MANIFEST.json'
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    def state(status,**extra):
        atomic(state_path,dict(status=status,controller_pid=os.getpid(),updated_beijing=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),**extra))
    with lock(folder/'CONTROLLER.lock'):
        try:
            if sha(__file__)!=policy['controller_sha256']:raise RuntimeError('controller identity mismatch')
            original=read(folder/'ORIGINAL_RUN_MANIFEST.json')
            if sha(folder/'ORIGINAL_RUN_MANIFEST.json')!=policy['original_manifest_sha256']:
                raise RuntimeError('original manifest changed')
            # Restart after a successful handoff must not re-evaluate evolving last.pt.
            handed=folder/'HANDOFF.json'
            if handed.exists():
                state('HANDED_OFF',handoff=read(handed),note='no duplicate launch on controller restart');return
            if sha(manifest_path)!=policy['original_manifest_sha256']:
                prepared=folder/'PREPARED_CONTINUATION_MANIFEST.json'
                if not prepared.exists() or sha(manifest_path)!=sha(prepared):raise RuntimeError('unexpected manifest mutation')
            check_source(original,policy)
            while True:
                if (folder/'CANCEL').exists():state('CANCELLED',continuation_started=False);return
                if (folder/'GATE_RESULT.json').exists() and not read(folder/'GATE_RESULT.json')['pass_gate']:
                    state('GATE_NOT_PASSED',continuation_started=False,gate=read(folder/'GATE_RESULT.json'));return
                if (root/'BOUNDED_RESULT.json').exists():break
                pipeline=read(root/'PIPELINE_STATE.json')
                if pipeline.get('status') in ('FAILED','STOPPED_INCOMPLETE'):
                    state('BLOCKED_INITIAL_STAGE_FAILED',pipeline=pipeline,continuation_started=False);return
                state('WAITING_FOR_COMPLETE_EPOCH5',current_epoch=pipeline.get('epoch',0),
                    current_update=pipeline.get('update',0),training_pid=pipeline.get('child_pid'))
                time.sleep(15)
            # Natural handoff: hold the run lock only after the old supervisor exits.
            with lock(root/'SUPERVISOR.lock',blocking=True):
                if (folder/'CANCEL').exists():state('CANCELLED',continuation_started=False);return
                check_source(original,policy)
                if sha(manifest_path) not in (policy['original_manifest_sha256'],sha(folder/'PREPARED_CONTINUATION_MANIFEST.json') if (folder/'PREPARED_CONTINUATION_MANIFEST.json').exists() else ''):
                    raise RuntimeError('manifest changed during wait')
                training=Path(original['training_output'])
                bounded=read(root/'BOUNDED_RESULT.json')
                if bounded['status']!='BOUNDED_TRAINING_COMPLETE' or bounded['completed_epochs']!=5:
                    raise RuntimeError('not the complete epoch5 boundary')
                if sha(training/'last.pt')!=bounded['checkpoint_sha256'] or sha(bounded['checkpoint'])!=bounded['checkpoint_sha256']:
                    raise RuntimeError('epoch5 complete state identity mismatch')
                identity=read(training/'RUN_IDENTITY.json')
                if identity['source_commit']!=policy['source_commit'] or identity['epochs']!=100 or identity['rollout'] or identity['projection']:
                    raise RuntimeError('training contract changed')
                validations=[read(training/f'validation/epoch_{e:03d}.json') for e in (1,2,5)]
                gate=decide(*validations)
                gate.update(checkpoint=str(training/'last.pt'),checkpoint_sha256=bounded['checkpoint_sha256'],
                    policy_sha256=sha(folder/'POLICY.json'),validation_sha256={str(e):sha(training/f'validation/epoch_{e:03d}.json') for e in (1,2,5)})
                atomic(folder/'GATE_RESULT.json',gate)
                if not gate['pass_gate']:
                    state('GATE_NOT_PASSED',continuation_started=False,gate=gate);return
                # Verify all metadata in the full state. Existing trainer performs
                # strict model + Adam restoration and restores all RNG on launch.
                import torch
                payload=torch.load(training/'last.pt',map_location='cpu',weights_only=False)
                if payload['epoch']!=5 or payload['run_identity']!=identity or payload['update_index']!=bounded['updates']:
                    raise RuntimeError('full checkpoint epoch/identity mismatch')
                scheduler=payload['scheduler_state']
                if scheduler['total_epochs']!=100 or scheduler['completed_epoch']!=5 or not payload['optimizer']['state'] or not payload['rng']:
                    raise RuntimeError('incomplete optimizer/scheduler/RNG state')
                del payload
                import gc;gc.collect()
                new=continuation_manifest(original,folder/'GATE_RESULT.json')
                release=read(root/'identity/RELEASE_IDENTITY.json')
                release.update(epochs_to_execute=100,resume_complete_epoch=5,
                    authorized_remaining_epochs=95,stop_rule='total_epoch100_then_existing_selector_and_five_inference_seeds',
                    continuation_policy_sha256=sha(folder/'POLICY.json'),continuation_gate_sha256=sha(folder/'GATE_RESULT.json'))
                atomic(folder/'CONTINUATION_RELEASE_IDENTITY.json',release)
                new['release_manifest_sha256']=sha(folder/'CONTINUATION_RELEASE_IDENTITY.json')
                new['continuation_release_identity']=str(folder/'CONTINUATION_RELEASE_IDENTITY.json')
                for path in (folder/'CONTINUATION_RELEASE_IDENTITY.json',folder/'POLICY.json',Path(__file__)):
                    new['launch_file_hashes'][str(path)]=sha(path)
                atomic(folder/'PREPARED_CONTINUATION_MANIFEST.json',new)
                atomic(manifest_path,new)
                state('HANDOFF_PREPARED',resume_from_epoch=5,target_total_epochs=100,gate=gate)
            # Same formal supervisor and GPU lock; it appends --resume and never
            # spawns two GPU processes. No source/config/checkpoint changes here.
            paper=Path(original['release_directory'])
            command=[original['command'][0],'-u',str(paper/'scripts/run_gated_release_background.py'),'--root',str(root)]
            env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONUNBUFFERED':'1'}
            with (root/'logs/supervisor_continue.log').open('a') as log:
                child=subprocess.Popen(command,cwd=paper,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=True,env=env)
            record=dict(supervisor_pid=child.pid,command=command,resume_complete_epoch=5,total_epochs=100,
                original_source_commit=policy['source_commit'],manifest_sha256=sha(manifest_path),
                checkpoint_sha256=gate['checkpoint_sha256'],beijing=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
            atomic(handed,record)
            for _ in range(30):
                if child.poll() is not None:
                    state('HANDOFF_CHILD_EXITED',returncode=child.returncode,handoff=record);return
                pipeline=read(root/'PIPELINE_STATE.json')
                if pipeline.get('supervisor_pid')==child.pid:
                    state('HANDED_OFF',handoff=record,pipeline=pipeline);return
                time.sleep(2)
            state('HANDOFF_STARTED_CHECK_PIPELINE',handoff=record)
        except BaseException as exc:
            state('FAILED',error=f'{type(exc).__name__}: {exc}',no_new_training_requested_after_failure=True)
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test()
    elif a.root:main(a.root)
    else:p.error('--root or --self-test required')
