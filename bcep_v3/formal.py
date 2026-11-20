"""Single ZARA2 Stage2 controller. Prepare is zero-update; train needs explicit opt-in."""
import argparse,contextlib,fcntl,os,shutil,subprocess,time
from pathlib import Path
import torch
from bcep.contracts import canonical_hash,digest
from bcep_joint.common import read,write,seed_all
from bcep_formal.control import install
from bcep_v2.state import run_stage
from bcep_v2.runtime import configure_cuda
from .formal_inputs import FormalInputs,paths,SOURCE,BASE,EXPERIMENT
from .formal_validation import validation
from .formal_telemetry import ModuleTelemetry
from .model import build,tensor_sha
from .stage2 import formal_callbacks


def model_hash(model,old_only=False):
    from .model import NEW_PREFIXES
    return canonical_hash({n:tensor_sha(t) for n,t in model.state_dict().items() if not old_only or not n.startswith(NEW_PREFIXES)})


def source_manifest():
    return {str(p.relative_to(SOURCE)):digest(p) for pkg in ['bcep','ebjd','bcep_v2','bcep_joint','bcep_formal','bcep_confirmation','selector_control','bcep_v3'] for p in sorted((SOURCE/pkg).rglob('*.py'))}


@contextlib.contextmanager
def locks(roots):
    roots['run'].mkdir(parents=True,exist_ok=True)
    with contextlib.ExitStack() as stack:
        for path,mode in [(roots['run']/'run.lock','a'),(roots['formal']/'controller.lock','r'),
                          (roots['candidate_cost']/'trial.lock','r'),(roots['sharedworld']/'run.lock','r')]:
            f=stack.enter_context(path.open(mode));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def freeze(inputs):
    seed_all(3201);model,origin=build(inputs.config)
    manifest=source_manifest()
    frozen=dict(schema='bcep-v3-zara2-stage2-ready-v1',base_source=BASE,experiment=EXPERIMENT,
                allow_real_training=False,real_optimizer_updates=0,auto_queue=False,config=inputs.config,recipe=inputs.recipe,
                implementation_files=manifest,implementation_hash=canonical_hash(manifest),provider=inputs.provider,
                split_sha256=canonical_hash(inputs.split),bank_sha256={b.purpose:b.hash for b in inputs.banks},counts=inputs.counts,
                initial_model_sha256=model_hash(model),initial_v2_subset_sha256=model_hash(model,True),
                max_real_updates_if_authorized=16800,candidate_seed=2035,training_seed=3201,
                selector='selection_dev H-JADE; exact ties earlier epoch',retention='all completed epochs, best, last, Top5 distinct epochs',
                runtime=dict(precision='FP32',deterministic=True,SDPA='math',TF32=False),
                module_telemetry='preclip gradient norms and actual parameter deltas after every optimizer step',
                activation='train requires RUN_ONE_ZARA2_V3=true and exact frozen configuration SHA; no launch by prepare/readiness')
    frozen['identity']=dict(stage='joint',fold='zara2',experiment=EXPERIMENT,base=BASE,
                            implementation=frozen['implementation_hash'],config=canonical_hash(inputs.config),recipe=canonical_hash(inputs.recipe),
                            split=frozen['split_sha256'],banks=frozen['bank_sha256'],initial=frozen['initial_model_sha256'])
    return frozen,origin


def require_authorization(command,frozen_sha):
    if command=='train' and (os.environ.get('RUN_ONE_ZARA2_V3')!='true' or not frozen_sha):
        raise PermissionError('No training authorized: require RUN_ONE_ZARA2_V3=true and --frozen-sha')


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train','status']);p.add_argument('--frozen-sha');a=p.parse_args()
    roots=paths();out=roots['run']
    if a.command=='status':print(read(out/'STATUS.json'));return
    require_authorization(a.command,a.frozen_sha) # Before CUDA, data loading, output creation, or model construction.
    with locks(roots):
        inputs=FormalInputs();frozen,origin=freeze(inputs)
        if a.command=='prepare':
            if (out/'joint/last.pt').exists():raise ValueError('cannot reprepare a started run')
            if (out/'FROZEN_CONFIG.json').exists() and read(out/'FROZEN_CONFIG.json')!=frozen:raise ValueError('frozen preparation changed')
            write(out/'FROZEN_CONFIG.json',frozen);write(out/'INITIALIZATION_ORIGIN.json',origin)
            write(out/'STATUS.json',dict(status='PREPARED_NO_TRAINING_AUTHORIZATION',optimizer_updates=0,allow_real_training=False))
            print('PREPARED',canonical_hash(frozen));return
        prior=read(out/'FROZEN_CONFIG.json')
        if prior!=frozen or a.frozen_sha!=canonical_hash(prior):raise ValueError('frozen source/config/initialization identity mismatch')
        ready=read(roots['audit']/'READINESS.json')
        if ready['status']!='PASS' or ready['frozen_sha256']!=a.frozen_sha or ready['real_optimizer_updates']!=0:raise ValueError('matching readiness acceptance required')
        configure_cuda();torch.set_num_threads(1);install();seed_all(3201)
        model,_=build(inputs.config);assert model_hash(model)==frozen['initial_model_sha256'];model.cuda()
        resume=(out/'joint/last.pt').exists();epoch=1;count=0
        if resume:
            saved=torch.load(out/'joint/last.pt',map_location='cpu',weights_only=False)
            if saved['identity']!=frozen['identity'] or saved['recipe']!=inputs.recipe:raise ValueError('recovery identity mismatch')
            epoch=saved['epoch'];count=saved['update']
            if saved['status']=='COMPLETE':print('already COMPLETE',count);return
            del saved
        # Conservative space for all remaining full snapshots plus atomic last/best.
        checkpoint_size=(roots['formal']/'zara2/joint/epoch_001.pt').stat().st_size*1.10
        if shutil.disk_usage(out).free < checkpoint_size*(103-epoch):raise RuntimeError('insufficient space for full checkpoint retention')
        native_backward,_=formal_callbacks(model,inputs,'cuda')
        observer=ModuleTelemetry(model,out/'MODULE_UPDATES.jsonl',count,'real')
        def backward(*args):
            result=native_backward(*args);observer.capture_backward();return result
        validate=validation(model,inputs.data,inputs.banks[1],out,epoch)
        write(out/'AUTHORIZATION.json',dict(RUN_ONE_ZARA2_V3=True,frozen_sha256=a.frozen_sha,unix=time.time(),single_fold='zara2',no_followup=True))
        write(out/'STATUS.json',dict(status='RUNNING',pid=os.getpid(),optimizer_updates=count,resume=resume))
        try:
            with observer.native_optimizer_hook():
                result=run_stage(out/'joint',frozen['identity'],inputs.recipe,model,inputs.split['train_indices'],backward,validate,16800,resume=resume,record_progress=True)
            status='PAUSED_RESUMABLE' if result['status']=='SESSION_BUDGET_PAUSED' else result['status']
            write(out/'STATUS.json',dict(status=status,optimizer_updates=result['update'],completed_epoch=result['completed_epoch'],stop_reason=result['stop_reason']))
        except BaseException as e:
            write(out/'STATUS.json',dict(status='TECHNICAL_INTERRUPTION',reason=str(e),exception_type=type(e).__name__));raise
if __name__=='__main__':main()
