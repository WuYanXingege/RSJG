"""Original loss/compaction weights, new provenance and resumable bounded runner."""
import sys,time,random
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_zara1_unified_coupling_bridge import bridge_and_pair_loss
from ebjd.data import augment_batch_with_points
from ebjd.metrics import MetricAccumulator
from .common import read,write,save,seed_all,rng_state,restore_rng,state_digest,digest,canonical_hash
from .model import build,optimizer_for,set_lr
from .inference import generate

class BudgetEnd(Exception):pass

def train_logical(model,opt,cpu_batch,proposals,gen,device,accumulation):
    batch,proposals=augment_batch_with_points(cpu_batch,proposals,gen)
    bs=len(batch.observed);agents=int(batch.valid.sum());time_draw=torch.randint(1,101,(bs,),device=device).float()/100
    noise=torch.randn(bs,1,batch.valid.shape[1],12,2,device=device,dtype=torch.float32)
    totals=[0.,0.,0.]
    for scene in range(bs):
        pos=batch.valid[scene].nonzero().flatten();physical=batch.compact_scenes(scene,scene+1).to(device)
        goals=proposals[scene:scene+1,:,pos].to(device)
        _,bridge,pair,agent_part,scene_part=bridge_and_pair_loss(model,physical,goals,time_draw=time_draw[scene:scene+1],noise=noise[scene:scene+1,:,pos.to(device)])
        total=len(pos)/agents*agent_part+scene_part/bs
        if not torch.isfinite(total):raise FloatingPointError('nonfinite joint loss')
        (total/accumulation).backward()
        totals[0]+=float(total.detach());totals[1]+=float(pair.detach())/bs;totals[2]+=float(bridge.detach())/bs
    return totals

def commit_step(model,opt):
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise FloatingPointError('nonfinite gradients')
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
    if not torch.isfinite(norm):raise FloatingPointError('nonfinite norm')
    opt.step();opt.zero_grad(set_to_none=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('nonfinite parameters')
    return float(norm)

def run(c,split,data,train_bank,dev_bank,identity,output,device='cuda',resume=False,stop_after_microbatches=None,model_factory=build):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);started=time.monotonic();base=0.
    prior=None
    if resume:
        prior=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
        assert prior['identity']==identity and prior['config_hash']==canonical_hash(c)
        base=max(prior['elapsed_seconds'],read(output/'STATUS.json')['elapsed_seconds'])
        assert prior['status'] not in ('COMPLETE','BUDGET_EXHAUSTED','NUMERICAL_OR_RUNTIME_FAILURE')
    seed_all(c['downstream_seed']);model=model_factory(c).to(device);opt=optimizer_for(model,c);gen=torch.Generator().manual_seed(c['downstream_seed'])
    init=dict(model=state_digest(model.state_dict()),components={n:state_digest(m.state_dict()) for n,m in [('map_unet',model.bridge.encoder.map_unet),('shared_encoder',model.bridge.encoder),('sixlayer_denoiser',model.bridge.denoiser),('coupling',model.coupling)]},optimizer_state_empty=not opt.state,optimizer_initial_sha256=canonical_hash(opt.state_dict()),pretrained_downstream_reads=0,loaded_learned_downstream_weights=[],initialization_role='random_all_downstream')
    if not resume:write(output/'INITIALIZATION_ORIGIN.json',init)
    state=dict(schema='bcep-sixlayer-joint-state-v1',identity=identity,config_hash=canonical_hash(c),epoch=1,completed_epoch=0,training_completed_epoch=0,phase='train',update=0,microbatch=0,order=None,cursor=0,pending=0,loss_sums=[0.,0.,0.],loss_count=0,dev_cursor=0,dev_acc=None,selection=dict(best=None,history=[]),scheduler=set_lr(opt,1),status='RUNNING')
    if prior:
        model.load_state_dict(prior['model'],strict=True);opt.load_state_dict(prior['optimizer'])
        for st in opt.state.values():
            for k,v in st.items():
                if torch.is_tensor(v):st[k]=v.to(device)
        for name,p in model.named_parameters():p.grad=prior['pending_gradients'][name].to(device) if name in prior['pending_gradients'] else None
        state={k:v for k,v in prior.items() if k not in ['model','optimizer','rng','pending_gradients']};restore_rng(prior['rng'],gen)
    def elapsed():return base+time.monotonic()-started
    def persist(reason,best=False):
        state.update(status=reason,elapsed_seconds=elapsed(),model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(gen),pending_gradients={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None})
        save(state,output/'last.pt')
        if best:save(state,output/'best.pt')
        write(output/'STATUS.json',{k:state[k] for k in ['status','epoch','completed_epoch','training_completed_epoch','update','microbatch','phase','cursor','pending','dev_cursor','selection'] }|dict(elapsed_seconds=elapsed(),checkpoint_sha256=digest(output/'last.pt')))
    invocation_micro=state['microbatch']
    try:
        while state['phase']!='complete':
            if (state['phase']=='validation' or state['pending']==0) and elapsed()>=c['budget_seconds']:
                persist('BUDGET_EXHAUSTED');break
            if stop_after_microbatches is not None and state['microbatch']-invocation_micro>=stop_after_microbatches:
                persist('TEST_INTERRUPTION');break
            if state['phase']=='train':
                model.train()
                if state['order'] is None:
                    state['order']=[split['train_indices'][i] for i in torch.randperm(len(split['train_indices']),generator=gen).tolist()]
                    state['scheduler']=set_lr(opt,state['epoch'])
                    order_file=output/'ORDER_RECEIPT.json';orders=read(order_file) if order_file.exists() else []
                    orders.append(dict(epoch=state['epoch'],scene_order_sha256=canonical_hash(state['order']),augmentation_generator_state=state_digest({'rng':gen.get_state()}),cpu_random_state=state_digest({'rng':torch.get_rng_state()}),cuda_random_state=state_digest({str(i):s for i,s in enumerate(torch.cuda.get_rng_state_all())}) if device!='cpu' else None))
                    write(order_file,orders)
                ids=state['order'][state['cursor']:state['cursor']+c['batch_scenes']]
                batch=data.batch(ids,'train');proposals=train_bank.batch(batch)
                values=train_logical(model,opt,batch,proposals,gen,device,c['gradient_accumulation'])
                state['loss_sums']=[a+b for a,b in zip(state['loss_sums'],values)];state['loss_count']+=1;state['cursor']+=len(ids);state['pending']+=1;state['microbatch']+=1
                last=state['cursor']==len(state['order'])
                if state['pending']==c['gradient_accumulation'] or last:
                    norm=commit_step(model,opt);state['update']+=1;state['pending']=0
                    with (output/'train.jsonl').open('a') as f:f.write(__import__('json').dumps(dict(epoch=state['epoch'],update=state['update'],microbatch=state['microbatch'],loss=values[0],pair_loss=values[1],gradient_norm=norm,lr=opt.param_groups[0]['lr'],rollout=0,projection=0,elapsed_seconds=elapsed()))+'\n')
                    if state['update']%10==0:write(output/'PROGRESS.json',dict(epoch=state['epoch'],update=state['update'],microbatch=state['microbatch'],elapsed_seconds=elapsed()))
                if last:state['training_completed_epoch']=state['epoch'];state['phase']='validation';state['dev_acc']={'G':MetricAccumulator().__dict__,'H':MetricAccumulator().__dict__}
            else:
                i=split['selection_dev_indices'][state['dev_cursor']];batch=data.batch([i],'selection_dev');goals=dev_bank.batch(batch).to(device);b=batch.to(device)
                g,h,_,_=generate(model,b.observed,b.semantic_maps,b.valid,b.scene_ids,b.agent_ids,goals,2035)
                for name,path in [('G',g),('H',h)]:
                    a=MetricAccumulator();a.__dict__.update(state['dev_acc'][name]);a.update(path,b.future,b.valid);state['dev_acc'][name]=a.__dict__
                state['dev_cursor']+=1
                if state['dev_cursor']<len(split['selection_dev_indices']):continue
                metrics={}
                for name in ['G','H']:
                    a=MetricAccumulator();a.__dict__.update(state['dev_acc'][name]);metrics[name]=a.as_dict()
                row=dict(epoch=state['epoch'],update=state['update'],loss=state['loss_sums'][0]/state['loss_count'],pair_loss=state['loss_sums'][1]/state['loss_count'],metrics=metrics,elapsed_seconds=elapsed())
                best=state['selection']['best'];improved=best is None or metrics['H']['JADE']<best['H_JADE']
                if improved:state['selection']['best']=dict(epoch=state['epoch'],update=state['update'],H_JADE=metrics['H']['JADE'])
                state['selection']['history'].append(row)
                with (output/'epochs.jsonl').open('a') as f:f.write(__import__('json').dumps(row)+'\n')
                state['completed_epoch']=state['epoch'];state['epoch']+=1;state.update(order=None,cursor=0,dev_cursor=0,dev_acc=None,loss_sums=[0.,0.,0.],loss_count=0)
                state['phase']='complete' if state['completed_epoch']>=c['max_epochs'] else 'train'
                persist('COMPLETE' if state['phase']=='complete' else 'EPOCH_COMPLETE',improved)
                print(f"{output.name} E{state['completed_epoch']:02d}/{c['max_epochs']} | loss={row['loss']:.5f} | H minADE/minFDE/JADE/JFDE="+'/'.join(f'{metrics["H"][k]:.5f}' for k in ['minADE','minFDE','JADE','JFDE'])+f" | updates={state['update']} wall={elapsed()/60:.1f}min",flush=True)
    except BaseException:
        persist('NUMERICAL_OR_RUNTIME_FAILURE');raise
    return state
