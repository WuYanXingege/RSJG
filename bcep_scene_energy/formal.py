"""One source-bound ZARA2 Energy-only Stage2. Preparation never starts training."""
import argparse,os,time
from .formal_contract import paths,require_authorization,locks,freeze,model_hash,file_sha,READY


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train','status']);p.add_argument('--frozen-sha');a=p.parse_args(argv)
    roots=paths();out=roots['run']
    if a.command=='status':
        import json
        print(json.loads((out/'STATUS.json').read_text()));return
    prior=require_authorization(a.command,a.frozen_sha,roots) # BEFORE data/model/optimizer/output creation.
    from bcep_joint.common import read,write,seed_all
    from bcep_v3.formal_inputs import FormalInputs
    with locks(roots):
        if read(roots['v3']/'STATUS.json')['status']!='COMPLETE':raise ValueError('protected v3 state changed')
        inputs=FormalInputs();frozen,origin=freeze(inputs)
        if a.command=='prepare':
            if (out/'joint/last.pt').exists():raise ValueError('cannot reprepare a started run')
            if (out/'FROZEN_CONFIG.json').exists() and read(out/'FROZEN_CONFIG.json')!=frozen:raise ValueError('immutable preparation differs')
            write(out/'FROZEN_CONFIG.json',frozen);write(out/'INITIALIZATION_ORIGIN.json',origin)
            write(out/'STATUS.json',dict(status='PREPARED_NO_TRAINING_AUTHORIZATION',optimizer_updates=0,allow_real_training=False))
            print('PREPARED file SHA256',file_sha(out/'FROZEN_CONFIG.json'));return
        if prior!=frozen:raise ValueError('verified current source/data/initialization differs from frozen identity')
        import torch,shutil
        from bcep_v2.runtime import configure_cuda
        from bcep_v2.state import run_stage
        from bcep_formal.control import install
        from .model import build
        from .formal_callbacks import formal_callbacks
        from .formal_validation import validation
        from .formal_monitor import StepObserver
        configure_cuda();torch.set_num_threads(1);install();seed_all(3201)
        model,_=build(inputs.config,variant='energy_only');assert model_hash(model)==frozen['initial_model_sha256'];model.cuda()
        resume=(out/'joint/last.pt').exists();epoch=1;updates=0
        if resume:
            saved=torch.load(out/'joint/last.pt',map_location='cpu',weights_only=False)
            if saved['identity']!=frozen['identity'] or saved['recipe']!=inputs.recipe:raise ValueError('resume identity mismatch')
            epoch=saved['epoch'];updates=saved['update']
            if saved['status']=='COMPLETE':print('already COMPLETE',updates);return
            del saved
        size=(roots['formal']/'zara2/joint/epoch_001.pt').stat().st_size*1.15
        if shutil.disk_usage(out).free<size*(103-epoch):raise RuntimeError('insufficient space to retain all complete epochs and atomic checkpoints')
        observer=StepObserver(model,out/'MODULE_UPDATES.jsonl',updates)
        backward=formal_callbacks(model,inputs,'cuda',observer)
        validate=validation(model,inputs,out,epoch,observer)
        write(out/'AUTHORIZATION.json',dict(RUN_ONE_ZARA2_SSEC=True,frozen_file_sha256=a.frozen_sha,unix=time.time(),single_fold='zara2',no_followup=True))
        write(out/'STATUS.json',dict(status='RUNNING',pid=os.getpid(),optimizer_updates=updates,resume=resume))
        try:
            with observer.native_optimizer_hook():
                result=run_stage(out/'joint',frozen['identity'],inputs.recipe,model,inputs.split['train_indices'],backward,validate,16800,resume=resume,record_progress=True)
            status='PAUSED_RESUMABLE' if result['status']=='SESSION_BUDGET_PAUSED' else result['status']
            write(out/'STATUS.json',dict(status=status,optimizer_updates=result['update'],completed_epoch=result['completed_epoch'],stop_reason=result['stop_reason']))
        except BaseException as e:
            write(out/'STATUS.json',dict(status='TECHNICAL_INTERRUPTION',reason=str(e),exception_type=type(e).__name__));raise
if __name__=='__main__':main()
