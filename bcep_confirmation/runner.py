"""Adapted frozen runner: factory, four-purpose access, initial hashes, explicit exception persistence.
Training objective, weighting, Adam/scheduler, RNG, cursors and selector are unchanged.
"""
import json,os,random,time
from pathlib import Path
import numpy as np
import torch
from bcep.provider import BCEP
from bcep_control.mdn import MDNProvider
from bcep_confirmation.common import state_digest,write
from bcep.contracts import canonical_hash,check_item,digest,require
from bcep.supervision import LikelihoodAccumulator,SELECTION_METRIC


def atomic_save(value,path):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temporary);os.replace(temporary,path)


def atomic_json(value,path):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2));os.replace(temporary,path)


def rng_state(generator,device):
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],loader=generator.get_state())


def restore_rng(state,generator,device):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['cpu']);generator.set_state(state['loader'])
    if device.type=='cuda':torch.cuda.set_rng_state_all(state['cuda'])


def run_pilot(dataset,split,cfg,identity,output,resume=False,stop_after_updates=None,on_update=None,workflow_started=None):
    """Same runner for the guarded CLI and deterministic synthetic resume tests.

    stop_after_updates is a test interruption, not a changed training contract.
    No completed epoch / scheduler advancement is invented at an interruption.
    """
    started=time.monotonic() if workflow_started is None else workflow_started;output=Path(output);output.mkdir(parents=True,exist_ok=True)
    last=output/'last.pt';receipt=output/'RUN_STATE.json';device=torch.device(cfg['device'])
    require(cfg['max_updates']<=len(split['train_indices'])*cfg['max_epochs'],'update cap exceeds locked epoch maximum')
    require(cfg['max_updates']>0 and cfg['budget_seconds']>0,'positive budgets required')
    require(cfg.get('selection_metric',SELECTION_METRIC)==SELECTION_METRIC,'unsupported selector')
    require(cfg.get('support',256)==256 and cfg.get('candidates',20)==20,'fixed support/candidate protocol differs')
    configuration_hash=canonical_hash(cfg)
    prior=None;base_elapsed=0.
    if resume:
        prior=torch.load(last,map_location='cpu',weights_only=False)
        require(prior['identity']==identity and prior['config_hash']==configuration_hash,'resume identity/config mismatch')
        base_elapsed=prior['elapsed_wall_seconds']
        if receipt.exists():
            r=json.loads(receipt.read_text())
            if r['checkpoint_sha256']==digest(last):base_elapsed=max(base_elapsed,r['elapsed_wall_seconds'])
        # No model construction, shuffle or update when an already exhausted run resumes.
        if base_elapsed>=cfg['budget_seconds'] or (prior['update']>=cfg['max_updates'] and prior['phase']=='train') or prior['phase']=='complete':return prior
    else:require(not last.exists(),'existing run requires explicit resume')
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    require(cfg['model_kind'] in ('bcep_flow','mdn8'),'invalid density')
    factory=BCEP if cfg['model_kind']=='bcep_flow' else MDNProvider
    model=factory(split['pred_length'],split['dt']).to(device)
    if not prior:
        write(output/'INITIALIZATION.json',dict(seed=cfg['seed'],model_kind=cfg['model_kind'],model_sha256=state_digest(model.state_dict()),encoder_sha256=state_digest(model.encoder.state_dict()),weights_origin='random_initialization',trained_weights_loaded=False))
    model.scales.copy_(torch.tensor(split['scales'],device=device));model.scales_fitted.fill_(True)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'])
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=cfg['lr_gamma'])
    generator=torch.Generator().manual_seed(cfg['seed'])
    state=dict(schema='bcep-confirmation-pilot-v1',identity=identity,config_hash=configuration_hash,
        protocol=dict(dt=split['dt'],pred_length=split['pred_length'],unit=split['unit']),
        completed_epoch=0,epoch=1,update=0,phase='train',order=None,train_cursor=0,dev_cursor=0,
        train_accumulator=None,dev_accumulator=None,best=None,stale=0,
        selection_metric_name=SELECTION_METRIC,selection_value=None,selected_epoch=None,split_hash=identity['split'])
    if prior:
        model.load_state_dict(prior['model'],strict=True);optimizer.load_state_dict(prior['optimizer']);scheduler.load_state_dict(prior['scheduler'])
        state={k:v for k,v in prior.items() if k not in ['model','optimizer','scheduler','rng']}
        restore_rng(prior['rng'],generator,device)
    invocation_start_update=state['update']
    def elapsed():return base_elapsed+time.monotonic()-started
    def save(reason,best=False):
        state.update(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                     rng=rng_state(generator,device),elapsed_wall_seconds=elapsed(),stop_reason=reason)
        atomic_save(state,last)
        if best:atomic_save(state,output/'best.pt')
        # Persist post-save/hash time so regular resumes do not reset persistence cost.
        sha=digest(last)
        atomic_json(dict(status=reason,checkpoint_sha256=sha,elapsed_wall_seconds=elapsed(),
            epoch=state['epoch'],completed_epoch=state['completed_epoch'],update=state['update'],
            phase=state['phase'],cursor=state['train_cursor'],budget_semantics='workflow wall time; one scene plus save overrun'),receipt)
    def one(index,train):
        item=dataset.get(index,'train' if train else 'selection_dev');check_item(item,split['members'][index])
        obs=item['observed'].to(device);maps=item.get('semantic_maps');maps=maps.to(device) if maps is not None else None
        valid=torch.ones(len(obs),dtype=torch.bool,device=device)
        with torch.set_grad_enabled(train):
            c=model.loss_components(obs,maps,valid,item['future'].to(device),cfg['lambda_aux'])
            require(bool(torch.isfinite(c['loss_total'])),'nonfinite loss')
            if train:
                optimizer.zero_grad(set_to_none=True);c['loss_total'].backward()
                require(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None),'nonfinite gradients')
                optimizer.step()
                require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),'nonfinite parameters')
        return c
    try:
        while state['phase']!='complete':
            if elapsed()>=cfg['budget_seconds']:save('BUDGET_EXHAUSTED');break
            if state['phase']=='train':
                if state['update']>=cfg['max_updates']:save('UPDATE_CAP');break
                if stop_after_updates is not None and state['update']-invocation_start_update>=stop_after_updates:
                    save('TEST_INTERRUPTION');break
                if state['order'] is None:
                    state['order']=[split['train_indices'][j] for j in torch.randperm(len(split['train_indices']),generator=generator).tolist()]
                model.train();index=state['order'][state['train_cursor']]
                c=one(index,True);acc=LikelihoodAccumulator(state['train_accumulator']);acc.update(c)
                state['train_accumulator']=acc.state;state['train_cursor']+=1;state['update']+=1
                if on_update:on_update(index,state['update'])
                if state['train_cursor']==len(state['order']):state['phase']='validation'
            else:
                model.eval();index=split['selection_dev_indices'][state['dev_cursor']]
                acc=LikelihoodAccumulator(state['dev_accumulator']);acc.update(one(index,False))
                state['dev_accumulator']=acc.state;state['dev_cursor']+=1
                if state['dev_cursor']<len(split['selection_dev_indices']):continue
                metrics=acc.metrics();value=metrics[SELECTION_METRIC]
                improved=state['best'] is None or value<state['best']
                if improved:state.update(best=value,selection_value=value,selected_epoch=state['epoch'])
                state['stale']=0 if improved else state['stale']+1
                scheduler.step();state['completed_epoch']=state['epoch']
                record=dict(epoch=state['epoch'],update=state['update'],elapsed_wall_seconds=elapsed())
                record.update(LikelihoodAccumulator(state['train_accumulator']).metrics('train'));record.update(metrics)
                with (output/'epochs.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print(f"Epoch {state['epoch']}/{cfg['max_epochs']} | dev final NLL(agent)={value:.6f} | updates={state['update']}",flush=True)
                state['epoch']+=1;state.update(order=None,train_cursor=0,dev_cursor=0,train_accumulator=None,dev_accumulator=None)
                state['phase']='complete' if state['completed_epoch']>=cfg['max_epochs'] or state['stale']>=cfg['patience'] else 'train'
                save('COMPLETE' if state['phase']=='complete' else 'EPOCH_COMPLETE',best=improved)
    except BaseException:
        save('INTERRUPTED_OR_NUMERICAL_ERROR')
        raise
    return state
