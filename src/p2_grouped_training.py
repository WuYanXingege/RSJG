"""New-family only fresh training; legacy trainers and objectives unchanged."""
import copy,json,math,random,re,time,subprocess
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from src.p2_protocol import (ROOT,Registry,FileProviders,digest,file_hash,path,data_binding,sealed,
    combine_train,BAD_BASES)
from src.p2_grouped import FAMILY,SCHEMA,require_provenance,qualification_issues
from src.p2_grouped_artifacts import atomic_tensor,PreparationGrant
from src.p2_observation import atomic_json
from src.p2_checkpoint import state_hash
from src.p2_data import base_packing_plan,slice_agents,join_agents,add_outer,observation_inputs,metric_inputs

class GeometryScene:
    """Only hash-bound H; no RGB, GT masks, raw source or cache constructor."""
    def __init__(self,name,ref):
        if file_hash(path(ref['path']))!=ref['sha256']:raise RuntimeError('geometry bytes')
        self.name=name;self.H=np.loadtxt(path(ref['path']));self.H_inv=np.linalg.inv(self.H)
        self.unit_of_measure='meter';self.frames_per_second=2.5;self.delta_frame=6 if name=='eth' else 10
        if self.H.shape!=(3,3) or not np.isfinite(self.H).all():raise RuntimeError('geometry matrix')
    def make_world_coord_torch(self,x):
        from src.data_src.data_utils import pixel2world_torch_multiagent
        x=x.clone()
        if self.name in ('eth','hotel'):x=x[...,[1,0]]
        return pixel2world_torch_multiagent(x,self.H)
    def make_pixel_coord_torch(self,x):
        from src.data_src.data_utils import world2pixel_torch_multiagent
        y=world2pixel_torch_multiagent(x.clone(),self.H_inv)
        return y[...,[1,0]] if self.name in ('eth','hotel') else y

def geometry_dataset(reg,roles=('train','inner_valid')):
    names={r['source']['scene_family'] for r in reg.records if r['role'] in roles}
    return SimpleNamespace(scenes={s:GeometryScene(s,reg.manifest['assets'][s]['H.txt']) for s in names})

class PackReader:
    """Production default; preparation explicitly accepts only its separate grant.

    inner metric provider is not invoked until the prediction callback returns.
    """
    def __init__(self,reg,role,*,preparation=None,provider=None):
        if role not in ('train','inner_valid'):raise PermissionError('outer base loader')
        if preparation is None:require_provenance(reg,(role,))
        elif not isinstance(preparation,PreparationGrant):raise TypeError('preparation grant required')
        self.reg=reg;self.role=role;self.preparation=preparation
        self.provider=provider or GroupedFileProvider(reg);self.packs=base_packing_plan(reg,role)
        self.read_trace=[]
    def __len__(self):return len(self.packs)
    def _read(self,i,kind):
        refs=self.packs[i];groups=defaultdict(list)
        # Role check for the ENTIRE pack before touching any provider.
        for wid,j in refs:
            r=self.reg.by_id[wid]
            if r['role']!=self.role:raise PermissionError('pack role changed')
            if self.preparation:
                purpose='observation_validation' if kind=='observation' else ('gradient_target_preparation' if self.role=='train' else 'metric_target_preparation')
                self.preparation.authorize(r,purpose)
            else:self.reg.authorize(wid,'gradient' if self.role=='train' else 'selection')
            groups[wid].append(j)
        pieces=[]
        for wid,indices in groups.items():
            self.read_trace.append((i,kind,wid))
            pieces.append(slice_agents(self.provider.read(self.reg.by_id[wid],kind),indices))
        value=join_agents(pieces)
        if kind=='observation':
            value['scene_index']=torch.zeros(len(refs),dtype=torch.long)
            value['scene_ptr']=torch.tensor([0,len(refs)],dtype=torch.long)
        return value
    def observation(self,i):return self._read(i,'observation')
    def training(self,i):
        if self.role!='train':raise PermissionError('gradient before provider')
        obs=self.observation(i);target=self._read(i,'target');data=combine_train(obs,target)
        refs=self.packs[i]
        data['frame_ids']=torch.stack([torch.tensor(self.reg.by_id[w]['source']['frame_ids']) for w,j in refs],1)
        data['abs_pixel_coord']=data['abs_pixel_coord']/8
        scene=self.reg.by_id[refs[0][0]]['source']['scene_family']
        return add_outer(data),{'scene_name':[scene],'p2_window_ids':[w for w,j in refs]},obs,target
    def predict_then_metric(self,i,predict,metric):
        if self.role!='inner_valid':raise PermissionError('selection only inner')
        obs=self.observation(i)
        with torch.no_grad():
            prediction=predict(obs)
            self.read_trace.append((i,'prediction_complete',None))
            target=self._read(i,'target')
            return metric(prediction,obs,target)

class GroupedFileProvider(FileProviders):
    def __init__(self,reg):super().__init__();self.reg=reg
    def read(self,r,kind):
        if kind=='target' and r['role']=='outer':raise PermissionError('outer target before provider')
        ref=r['artifacts'].get(kind)
        if ref and path(ref['path']).name!=r['window_id']+'.pt':raise RuntimeError('grouped artifact filename identity')
        if kind=='observation':
            meta=json.loads(path(ref['path']).with_suffix('.json').read_text())
            if sealed(meta)!=meta:raise RuntimeError('observation sidecar seal')
            if meta.get('schema')=='grouped-artifact-v1':
                original=path(meta['extra']['old_artifact']['path']).with_suffix('.json')
                if file_hash(original)!=meta['extra']['old_sidecar_sha256']:raise RuntimeError('observation original sidecar')
                meta=json.loads(original.read_text())
            if meta.get('asset_binding_hash')!=digest(self.reg.manifest['assets'][r['source']['scene_family']]):
                raise RuntimeError('observation assets differ from current input contract')
        if kind=='target':
            side=path(ref['path']).with_suffix('.json');m=json.loads(side.read_text())
            if sealed(m)!=m or m.get('role')!=r['role'] or m.get('source_identity')!=digest(r['source']) or m.get('sha256')!=ref['sha256']:raise RuntimeError('target sidecar role/identity/hash')
            expected='gradient_target_preparation' if r['role']=='train' else 'metric_target_preparation'
            if m.get('purpose')!=expected or m.get('frames')!=r['source']['frame_ids'][8:] or m.get('agents')!=r['source']['agent_ids']:raise RuntimeError('target purpose/slots')
            # Parent is the preparation manifest, not the later qualification review.
            if m.get('parent_manifest_hash')!=self.reg.manifest.get('preparation_parent_hash',self.reg.manifest['manifest_hash']):raise RuntimeError('target parent manifest')
        return super().read(r,kind)

def goal_bce_parts(logits,obs,target):
    if logits.ndim!=5 or logits.shape[0]!=1:raise RuntimeError('goal selection expects single logits sample')
    y=target['input_traj_maps'].to(logits.device,dtype=torch.float32)
    per=F.binary_cross_entropy_with_logits(logits[0].float(),y,reduction='none').mean((1,2,3))
    mask=torch.cat([obs['seq_list'],target['seq_list']],0).cumprod(0)[-1].to(per)
    return float((per*mask).double().sum().item()),float(mask.sum().item())

class WeightedMean:
    def __init__(self):self.numerator=0.;self.denominator=0.
    def add(self,n,d):
        if not math.isfinite(n) or not math.isfinite(d) or d<0:raise ValueError('invalid metric')
        self.numerator+=n;self.denominator+=d
    def value(self):
        if self.denominator<=0:raise ValueError('zero valid denominator')
        return self.numerator/self.denominator

class Selector:
    def __init__(self,min_delta=0.,patience=0):
        if min_delta<0 or patience<0:raise ValueError('selector settings')
        self.min_delta=min_delta;self.patience=patience;self.best=None;self.epoch=None;self.bad=0
    def update(self,value,epoch):
        if not math.isfinite(value):raise ValueError('nonfinite selection')
        improved=self.best is None or value<self.best-self.min_delta
        if improved:self.best=value;self.epoch=epoch;self.bad=0
        else:self.bad+=1
        return improved,bool(self.patience and self.bad>=self.patience)

class L2Selector:
    """Disabled by configuration; lexicographic JADE/JFDE, strict ties."""
    def __init__(self,min_delta=0.,patience=0):
        if min_delta<0 or patience<0:raise ValueError('L2 settings')
        self.delta=min_delta;self.patience=patience;self.best=None;self.epoch=None;self.bad=0
    def update(self,jade,jfde,epoch,role='inner_valid'):
        if role!='inner_valid':raise PermissionError('L2 selection inner only')
        if not all(math.isfinite(x) for x in (jade,jfde)):raise ValueError('nonfinite L2')
        good=self.best is None or jade<self.best[0]-self.delta or (jade==self.best[0] and jfde<self.best[1]-self.delta)
        if good:self.best=(jade,jfde);self.epoch=epoch;self.bad=0
        else:self.bad+=1
        return good,bool(self.patience and self.bad>=self.patience)

@torch.no_grad()
def l2_inner_evaluate(model,reg,device,provider=None):
    """Disabled-by-default future L2 adapter: native windows, not mixed packs.

    Targets are read only after observation-only joint prediction. Existing
    JADE/JFDE return one world-metric value per scene/window, in metres.
    Overlapping windows are not claimed to be independent recordings.
    """
    from src.metrics import compute_metric_mask
    require_provenance(reg,('inner_valid',))
    provider=provider or GroupedFileProvider(reg)
    sums={'JADE':WeightedMean(),'JFDE':WeightedMean()};model.eval()
    for r in reg.refs_unordered('inner_valid'):
        if r['role']!='inner_valid':raise PermissionError('L2 inner role before provider')
        reg.authorize(r['window_id'],'selection')
        obs=provider.read(r,'observation');scene=model.dataset.scenes[r['source']['scene_family']]
        pred,aux=model(observation_inputs(obs,scene,device),if_test=True)
        target=provider.read(r,'target')
        inputs=metric_inputs(obs,target,scene,device);mask=compute_metric_mask(inputs['seq_list'])
        inputs['x_augmented']=torch.zeros(*inputs['abs_pixel_coord'].shape[:2],8,device=device)
        inputs['x_augmented'][...,6:]=inputs['abs_pixel_coord']
        for name,acc in sums.items():
            values=model.compute_model_metrics(metric_name=name,predictions=pred,metric_mask=mask,
                all_aux_outputs=aux,inputs=inputs,obs_length=8)
            acc.add(sum(values),len(values))
    return {k:{'value':v.value(),'denominator':v.denominator,'units':'world_m',
        'aggregation':'native scene/window metric; overlapping windows not independent'} for k,v in sums.items()}

def seed_all(seed,device):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.device(device).type=='cuda':torch.cuda.manual_seed_all(seed)

def rng_snapshot(device):
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=[n[0],n[1].tolist(),n[2],n[3],n[4]],
        torch_cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.device(device).type=='cuda' else [],
        workers='num_workers=0; none',independent_mc='NOT_APPLICABLE_GOAL_BASE_MEAN_ENERGY')

def restore_rng(r):
    random.setstate(r['python']);n=r['numpy'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(r['torch_cpu'])
    if r['cuda']:torch.cuda.set_rng_state_all(r['cuda'])

def tree_hash(value):
    """State identity independent of CPU/CUDA storage location."""
    def normalize(x):
        if torch.is_tensor(x):return {'tensor':state_hash({'value':x})}
        if isinstance(x,dict):return {'dict':[(repr(k),normalize(v)) for k,v in sorted(x.items(),key=lambda kv:repr(kv[0]))]}
        if isinstance(x,(list,tuple)):return {type(x).__name__:[normalize(v) for v in x]}
        return x
    return digest(normalize(value))

def source_fingerprint():
    names=['src/p2_grouped.py','src/p2_grouped_training.py','src/p2_grouped_artifacts.py',
        'src/models/model.py','src/models/goal_pretrain.py','src/losses.py','src/p2_data.py','src/parser.py']
    return dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        files={n:file_hash(ROOT/n) for n in names},torch=str(torch.__version__),numpy=np.__version__)

def architecture(model):
    return dict(parameters=sum(p.numel() for p in model.parameters()),
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        shapes={k:list(v.shape) for k,v in model.state_dict().items()},
        modules={k:type(v).__name__ for k,v in model.named_modules()},
        goal_encoder=list(model.enc_chs),goal_decoder=list(model.dec_chs),
        layer_specs={k:{key:getattr(v,key) for key in ('in_channels','out_channels','kernel_size','in_features','out_features','input_size','hidden_size','num_layers','batch_first','num_heads','dropout','p')
            if hasattr(v,key) and isinstance(getattr(v,key),(int,float,bool,tuple))} for k,v in model.named_modules()})

def save_checkpoint(dest,model,opt,sched,reg,args,progress,initial,classification,metric=None):
    if classification not in ('SMOKE_ONLY','FORMAL_EPOCH'):raise ValueError('checkpoint classification')
    if classification=='FORMAL_EPOCH' and (progress['cursor']!=0 or progress['epoch_completed']<1 or sched.last_epoch!=progress['epoch_completed']):raise RuntimeError('complete epoch after scheduler required')
    arc=architecture(model)
    payload=dict(schema='grouped-training-state-v1',classification=classification,model_state_dict=model.state_dict(),source=source_fingerprint(),
        optimizer=opt.state_dict(),scheduler=sched.state_dict(),rng=rng_snapshot(args.device),progress=progress,
        data_binding=data_binding(reg),family=FAMILY,manifest_hash=reg.manifest['manifest_hash'],architecture=arc,
        architecture_hash=digest(arc),initial_state_sha256=initial,config=vars(args),config_hash=digest(vars(args)),
        state_sha256=state_hash(model.state_dict()),gradient_roles=['train'],selection_role='inner_valid',outer_access=False,
        metric=metric,artifact_role='goal' if args.p2_mode=='goal' else 'base',resume_policy='FRESH_START_ONLY',
        boundary='complete_epoch_post_scheduler' if classification=='FORMAL_EPOCH' else 'smoke_optimizer_boundary')
    payload.update(optimizer_hash=tree_hash(payload['optimizer']),scheduler_hash=tree_hash(payload['scheduler']),rng_hash=tree_hash(payload['rng']))
    if classification=='FORMAL_EPOCH':
        init_path=Path(dest).parent/'initial.pt'
        init=torch.load(init_path,map_location='cpu',weights_only=True)
        if init.get('optimizer_updates')!=0 or state_hash(init['model_state_dict'])!=initial:raise RuntimeError('formal initial snapshot')
        payload['initial_ref']=dict(path=str(init_path),sha256=file_hash(init_path))
        payload['train_packing_hash']=digest(base_packing_plan(reg,'train'))
        payload['inner_packing_hash']=digest(base_packing_plan(reg,'inner_valid'))
    atomic_tensor(payload,dest);actual=file_hash(dest)
    check=torch.load(dest,map_location='cpu',weights_only=True)
    if state_hash(check['model_state_dict'])!=payload['state_sha256'] or check['progress']!=progress:raise RuntimeError('checkpoint readback')
    if any(tree_hash(check[k])!=payload[k+'_hash'] for k in ('optimizer','scheduler','rng')):raise RuntimeError('checkpoint training-state readback')
    receipt=dict(path=str(dest),sha256=actual,state_sha256=payload['state_sha256'],classification=classification,
        architecture_hash=payload['architecture_hash'],data_binding=payload['data_binding'],progress=progress)
    atomic_json(receipt,str(dest)+'.json');return receipt

def qualify_parent(ref,reg,role):
    """Read-only validator; never auto-promotes smoke or incomplete weights."""
    if not ref or not ref.get('path'):raise RuntimeError('WAITING_FOR_QUALIFIED_'+role.upper())
    actual=file_hash(path(ref['path']))
    if actual in BAD_BASES or actual!=ref.get('sha256'):raise RuntimeError('parent actual hash or known violation')
    v=torch.load(path(ref['path']),map_location='cpu',weights_only=True)
    if v.get('schema')!='grouped-training-state-v1' or v.get('classification')!='FORMAL_EPOCH' or v.get('artifact_role')!=role:raise RuntimeError('parent classification/schema')
    if v.get('data_binding')!=data_binding(reg) or v.get('family')!=FAMILY:raise RuntimeError('parent protocol/assets/order')
    if v.get('gradient_roles')!=['train'] or v.get('selection_role')!='inner_valid' or v.get('outer_access') is not False:raise RuntimeError('parent exposures roles')
    p=v['progress'];epoch=p['epoch_completed']
    if not 1<=epoch<=({'goal':150,'base':250}[role]) or p['cursor']!=0 or p['successful']!=epoch*175 or p['attempts']!=p['successful'] or p['agent_exposures']!=epoch*11122 or p['skipped'] or p['failed']:raise RuntimeError('parent complete epoch/exposure')
    if v['scheduler']['last_epoch']!=epoch or v['boundary']!='complete_epoch_post_scheduler':raise RuntimeError('parent scheduler boundary')
    if not v['initial_state_sha256'] or digest(v['architecture'])!=v['architecture_hash'] or state_hash(v['model_state_dict'])!=v['state_sha256']:raise RuntimeError('parent state/architecture/init')
    metric=v.get('metric') or {}
    if metric.get('epoch')!=epoch or metric.get('denominator')!=24955 or not math.isfinite(metric.get('value',float('nan'))) or metric.get('tie')!='strict_earlier' or metric.get('is_best') is not True:raise RuntimeError('parent best selection')
    if v.get('train_packing_hash')!=digest(base_packing_plan(reg,'train')) or v.get('inner_packing_hash')!=digest(base_packing_plan(reg,'inner_valid')):raise RuntimeError('parent exposure packing')
    initref=v.get('initial_ref') or {}
    if not initref.get('path') or file_hash(path(initref['path']))!=initref.get('sha256'):raise RuntimeError('parent init file')
    init=torch.load(path(initref['path']),map_location='cpu',weights_only=True)
    if init.get('optimizer_updates')!=0 or init.get('seed')!=3101 or state_hash(init['model_state_dict'])!=v['initial_state_sha256']:raise RuntimeError('parent initial binding')
    if role=='base':
        gp=qualify_parent(init.get('goal_parent'),reg,'goal')
        selected={k:x for k,x in init['model_state_dict'].items() if k.startswith('goal_module.')}
        if state_hash(selected)!=state_hash(gp['model_state_dict']):raise RuntimeError('base initial goal-only parent')
        fresh={k:x for k,x in init['model_state_dict'].items() if not k.startswith('goal_module.')}
        if state_hash(fresh)!=init.get('fresh_non_goal_sha256'):raise RuntimeError('base fresh history/diffusion')
    if v.get('config_hash')!=digest(v['config']) or v['config']['seed']!=3101:raise RuntimeError('parent config')
    if any(tree_hash(v[k])!=v.get(k+'_hash') for k in ('optimizer','scheduler','rng')):raise RuntimeError('parent optimizer/scheduler/RNG identity')
    if {k:list(x.shape) for k,x in v['model_state_dict'].items()}!=v['architecture']['shapes']:raise RuntimeError('parent architecture shapes')
    history=p.get('selection_history',[])
    if len(history)!=epoch or [x['epoch'] for x in history]!=list(range(1,epoch+1)):raise RuntimeError('parent selection history')
    if min(history,key=lambda x:x['value'])['epoch']!=epoch:raise RuntimeError('parent strict earlier best')
    for x in history:
        if not math.isfinite(x['value']) or x['denominator']!=24955 or x['train_exposures']!=11122 or x['updates']!=175:raise RuntimeError('parent per-epoch audit')
    steps=[int(x['step']) for x in v['optimizer']['state'].values() if 'step' in x]
    if not steps or any(s!=p['successful'] for s in steps):raise RuntimeError('parent optimizer steps')
    lr=(.001 if role=='goal' else .0001)*(.99**epoch)
    if any(not math.isclose(g['lr'],lr,rel_tol=1e-12) for g in v['optimizer']['param_groups']):raise RuntimeError('parent scheduler LR')
    run_path=path(ref['path']).parent/'run_complete.json'
    if not run_path.is_file():raise RuntimeError('parent requires completed-run selection receipt')
    run=json.loads(run_path.read_text())
    if sealed(run)!=run or run.get('run_complete') is not True or run.get('data_binding')!=data_binding(reg):raise RuntimeError('parent run completion identity')
    if run.get('selected_sha256')!=actual or run.get('best_epoch')!=epoch or run.get('config_hash')!=v['config_hash']:raise RuntimeError('parent is not final selected checkpoint')
    full_history=run.get('selection_history',[])
    if len(full_history)!=run.get('complete_epochs') or full_history[:epoch]!=history or min(full_history,key=lambda x:x['value'])['epoch']!=epoch:raise RuntimeError('parent full-run strict selection')
    require_provenance(reg,('train','inner_valid'))
    return v

def formal_output(args):
    return path(Path('output')/str(args.test_set)/'runs'/args.run_name)

def launch_guard(args,reg):
    mode=args.p2_mode
    if mode in ('goal','joint'):
        require_provenance(reg,('train','inner_valid'))
        required=dict(goal_model_type='independent',training_stage='baseline',seed=3101,batch_size=64,
            num_workers=0,data_augmentation=False,coupling_grad_accum_steps=1,down_factor=8,
            final_test_access='blocked',optimizer='Adam',scheduler='ExponentialLR',learning_rate=.001 if mode=='goal' else .0001,
            clip=1.,amp_enabled=False,amp_dtype='fp32',validate_every=1,shuffle_train_batches=False,
            use_scene_latent=False,use_dynamic_relation=False,use_joint_energy=False,use_dependency_corrector=False)
        for k,v in required.items():
            if getattr(args,k,None)!=v:raise RuntimeError('grouped frozen option '+k)
        if args.phase!=('goal_pretrain' if mode=='goal' else 'train') or not 1<=args.num_epochs<=({'goal':150,'joint':250}[mode]):raise RuntimeError('grouped phase/epoch')
        if getattr(args,'pretrain_path',None) or getattr(args,'load_checkpoint',None):raise RuntimeError('FRESH_START_ONLY')
        if not args.run_name or not re.fullmatch(r'grouped_fresh_[A-Za-z0-9_.-]+',args.run_name):raise RuntimeError('independent formal namespace')
        dest=formal_output(args)
        if dest.exists():raise RuntimeError('formal output must not exist; FRESH_START_ONLY')
        if mode=='joint':qualify_parent(reg.manifest.get('parents',{}).get('goal'),reg,'goal')
        for r in reg.records:
            if r['role']=='outer':continue
            for kind in ('observation','target'):
                ref=r['artifacts'].get(kind)
                if not ref or not path(ref['path']).is_file() or file_hash(path(ref['path']))!=ref['sha256']:raise RuntimeError('missing/tampered grouped '+kind)
        return reg
    # Future cache/L1 retain existing guarded flow, with new schema dispatched there.
    require_provenance(reg,('train','inner_valid','outer') if mode=='cache' else ('train',))
    return None

@torch.no_grad()
def evaluate(model,reader,device,goal,deadline=None):
    from src.metrics import compute_metric_mask
    model.eval();acc=WeightedMean()
    for i,refs in enumerate(reader.packs):
        if deadline is not None and time.monotonic()>=deadline:
            raise RuntimeError('FORMAL_WALL_PARTIAL_SELECTION_NO_CHECKPOINT')
        scene=model.dataset.scenes[reader.reg.by_id[refs[0][0]]['source']['scene_family']]
        def predict(obs):
            x=observation_inputs(obs,scene,device)
            return model(x) if goal else model(x,if_test=True)
        def metric(pred,obs,target):
            if goal:return goal_bce_parts(pred,obs,target)
            inputs=metric_inputs(obs,target,scene,device);mask=compute_metric_mask(inputs['seq_list'])
            inputs['x_augmented']=torch.zeros(*inputs['abs_pixel_coord'].shape[:2],8,device=device)
            inputs['x_augmented'][...,6:]=inputs['abs_pixel_coord']
            values=model.compute_model_metrics(metric_name='ADE',predictions=pred[0],metric_mask=mask,all_aux_outputs=pred[1],inputs=inputs,obs_length=8)
            return sum(values),len(values)
        acc.add(*reader.predict_then_metric(i,predict,metric))
    return acc

def train(args,reg):
    """Only explicitly invoked formal path. Never used by preflight smoke."""
    from src.models.goal_pretrain import Goal_Pretrain
    from src.models.model import GDTS
    launch_guard(args,reg);goal=args.p2_mode=='goal';device=torch.device(args.device)
    seed_all(3101,device);dest=formal_output(args);dest.mkdir(parents=True,exist_ok=False)
    dataset=geometry_dataset(reg);model=(Goal_Pretrain if goal else GDTS)(args,device,dataset=dataset).to(device)
    non_goal=state_hash({k:v for k,v in model.state_dict().items() if not k.startswith('goal_module.')})
    if not goal:
        parent=qualify_parent(reg.manifest['parents']['goal'],reg,'goal')
        model.goal_module.load_state_dict({k[len('goal_module.'):]:v for k,v in parent['model_state_dict'].items() if k.startswith('goal_module.')},strict=True)
    if not goal and state_hash({k:v for k,v in model.state_dict().items() if not k.startswith('goal_module.')})!=non_goal:raise RuntimeError('goal load touched fresh history/diffusion')
    initial=state_hash(model.state_dict());atomic_tensor(dict(classification='FORMAL_INITIAL_UNUPDATED',state_sha256=initial,model_state_dict=model.state_dict(),optimizer_updates=0,seed=3101,
        goal_parent=None if goal else reg.manifest['parents']['goal'],fresh_non_goal_sha256=non_goal),dest/'initial.pt')
    opt=torch.optim.Adam(model.parameters(),lr=args.learning_rate);sched=torch.optim.lr_scheduler.ExponentialLR(opt,gamma=.99)
    tr=PackReader(reg,'train');va=PackReader(reg,'inner_valid');selector=Selector(patience=args.early_stopping_patience)
    p=dict(epoch_completed=0,cursor=0,attempts=0,successful=0,skipped=0,failed=0,agent_exposures=0,selection_history=[])
    start=time.monotonic();deadline=24*3600 if goal else 48*3600
    for epoch in range(1,args.num_epochs+1):
        model.train()
        for i in range(len(tr)):
            # Cap BEFORE next pack read. A partial epoch never selects/schedules/saves.
            if time.monotonic()-start>=deadline:raise RuntimeError('FORMAL_WALL_PARTIAL_EPOCH_NO_CHECKPOINT')
            batch,ids,_,_=tr.training(i);prepared=model.prepare_inputs(batch,ids);x=prepared[0];seq=prepared[-1]
            p['attempts']+=1;opt.zero_grad(set_to_none=True)
            losses=model.get_loss(x,seq);coeff=model.set_losses_coeffs();loss=sum(v*coeff[k] for k,v in losses.items())
            if not torch.isfinite(loss):p['failed']+=1;raise FloatingPointError('nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip,error_if_nonfinite=True);opt.step()
            p['successful']+=1;p['cursor']=i+1;p['agent_exposures']+=len(tr.packs[i])
        seed_snapshot=rng_snapshot(device);seed_all(args.validation_seed,device)
        try:metric=evaluate(model,va,device,goal,start+deadline)
        finally:restore_rng(seed_snapshot)
        if metric.denominator!=24955:raise RuntimeError('incomplete inner metric denominator')
        good,stop=selector.update(metric.value(),epoch);sched.step();p['epoch_completed']=epoch;p['cursor']=0
        meta=dict(value=metric.value(),denominator=metric.denominator,epoch=epoch,tie='strict_earlier',is_best=good,
            name='valid_goal_BCE_agent_weighted' if goal else 'valid_ADE_native_pixels',samples=1 if goal else args.num_samples)
        p['selection_history'].append(dict(epoch=epoch,value=metric.value(),denominator=metric.denominator,train_exposures=11122,updates=175))
        save_checkpoint(dest/('epoch_%03d.pt'%epoch),model,opt,sched,reg,args,dict(p),initial,'FORMAL_EPOCH',meta)
        atomic_json(dict(last_epoch=epoch,best_epoch=selector.epoch,best_value=selector.best,progress=p),dest/'selection.json')
        if stop:break
    best_path=dest/('epoch_%03d.pt'%selector.epoch)
    atomic_json(sealed(dict(run_complete=True,complete_epochs=p['epoch_completed'],best_epoch=selector.epoch,
        best_value=selector.best,selected_path=str(best_path),selected_sha256=file_hash(best_path),
        config_hash=digest(vars(args)),data_binding=data_binding(reg),selection_history=p['selection_history'],
        stop_reason='early_stop' if stop else 'epoch_cap',optimizer_attempts=p['attempts'],successful_updates=p['successful'])),dest/'run_complete.json')
    return dict(best_epoch=selector.epoch,best_value=selector.best)
