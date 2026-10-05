"""Role-indexed independent artifacts and observation-only input adapter."""
import copy
from collections import Counter
import torch
from src.p2_protocol import (read_record,FileProviders,load_registry,launch_guard,
    combine_train,digest,identity)

def single(value):return value

def add_outer(value):
    return {k:(v.unsqueeze(0) if torch.is_tensor(v) else add_outer(v) if isinstance(v,dict) else v)
            for k,v in value.items()}

def base_packing_plan(registry,role,batch_size=64):
    """Independent GDTS: retain every (window,agent), flush on source/map change."""
    packs=[];pending=[];source=None
    for r in registry.refs_unordered(role):
        s=r['source']['source_sha256']
        if pending and source!=s:packs.append(pending);pending=[]
        source=s
        for agent in range(len(r['source']['agent_ids'])):
            pending.append((r['window_id'],agent))
            if len(pending)==batch_size:packs.append(pending);pending=[]
    if pending:packs.append(pending)
    return packs

def slice_agents(value,indices):
    result={}
    for key,v in value.items():
        if key in {'abs_pixel_coord','seq_list','frame_ids'}:result[key]=v[:,indices]
        elif key=='input_traj_maps':result[key]=v[indices]
        elif key not in {'scene_index','scene_ptr'}:result[key]=v
    return result

def join_agents(parts):
    out=dict(parts[0])
    for key,dim in [('abs_pixel_coord',1),('seq_list',1),('frame_ids',1),('input_traj_maps',0)]:
        if key in out:out[key]=torch.cat([p[key] for p in parts],dim)
    return out

class RoleDataset:
    def __init__(self,registry,role,purpose,providers=None,*,base_pack=False):
        # Authorization BEFORE creating any provider or touching payload.
        self.registry=registry;self.role=role;self.purpose=purpose
        self.records=registry.refs(role)
        for r in self.records:registry.authorize(r['window_id'],purpose)
        self.providers=providers or FileProviders()
        self.packs=base_packing_plan(registry,role) if base_pack else None
        self.source_ids=[r['window_id'] for r in self.records]
        self.logical_set_name=role
        self.data_augmentation=False
        cache=registry.manifest.get('cache')
        self.jdv2_manifest=(dict(cache,manifest_hash=digest(cache),
            goal_checkpoint_hash=cache['producer_sha256']) if cache else None)
    def __len__(self):return len(self.packs) if self.packs is not None else len(self.records)
    def __getitem__(self,i):
        refs=self.packs[i] if self.packs is not None else [(self.records[i]['window_id'],None)]
        loaded={};observations=[];targets=[]
        for wid,agent in refs:
            if wid not in loaded:loaded[wid]=read_record(self.registry,wid,self.purpose,self.providers)
            sample=loaded[wid]
            observations.append(sample['observation'] if agent is None else slice_agents(sample['observation'],[agent]))
            if sample['target'] is not None:
                targets.append(sample['target'] if agent is None else slice_agents(sample['target'],[agent]))
        obs=join_agents(observations);target=join_agents(targets) if targets else None
        r=loaded[refs[0][0]]['ref']
        ids={'scene_name':[r['source']['scene_family']],
             'p2_window_ids':[wid for wid,_ in refs],
             'p2_identity_hash':digest(refs),'p2_role':self.role}
        if self.purpose in {'gradient','jdv2_gradient'}:
            data=combine_train(obs,target)
            # Full frames are metadata, not future coordinates.
            if self.packs is None:
                data['frame_ids']=torch.tensor(r['source']['frame_ids'])[:,None].repeat(1,len(r['source']['agent_ids']))
            if self.purpose=='jdv2_gradient':
                data['jdv2_cache']=self.providers.read(r,'deployment')
                data['jdv2_teacher_cache']=loaded[r['window_id']]['teacher']
            data['abs_pixel_coord']=data['abs_pixel_coord']/float(self.registry.manifest['down_factor'])
            return add_outer(data),ids
        return dict(observation=obs,target=target,identity=ids)

def get_p2_dataloader(args,set_name):
    reg=launch_guard(args)
    if reg.manifest.get('family')=='p2_grouped_univ_hotel_v1' and args.p2_mode in {'goal','joint'}:
        raise RuntimeError('Grouped base training uses the dedicated PackReader runner; legacy eager-target loader forbidden')
    if set_name=='test':raise PermissionError('P2 outer metrics locked')
    if args.p2_mode=='l1' and set_name!='train':raise PermissionError('P2 L1 no valid loader')
    role='train' if set_name=='train' else 'inner_valid'
    purpose=('jdv2_gradient' if args.p2_mode=='l1' else 'gradient') if role=='train' else 'selection'
    dataset=RoleDataset(reg,role,purpose,base_pack=args.p2_mode in {'goal','joint'})
    return torch.utils.data.DataLoader(dataset,batch_size=None,shuffle=False,num_workers=0,
                                      collate_fn=single)

def observation_inputs(observation,scene,device,down_factor=8):
    """No padded future, no full-payload prepare_inputs. Original observed derivatives."""
    from src.models.model import derivative_of
    inputs={k:v.to(device).float() if torch.is_tensor(v) else v for k,v in observation.items()}
    xy=inputs['abs_pixel_coord']/down_factor
    if xy.shape[0]!=8:raise RuntimeError('P2 observation adapter requires exactly obs8')
    inputs['abs_pixel_coord']=xy
    relative=xy-xy[-1]
    velocity=derivative_of(relative,dt=1)
    acceleration=derivative_of(velocity,dt=1)
    inputs['x_augmented']=torch.cat((relative,velocity,acceleration,xy),dim=-1)
    world=scene.make_world_coord_torch(xy*down_factor)
    inputs['world_coord']=world
    inputs['obs_traj_world']=world.permute(1,0,2).contiguous()
    inputs['scene']=scene
    n=xy.shape[1]
    inputs['scene_index']=torch.zeros(n,dtype=torch.long,device=device)
    inputs['scene_ptr']=torch.tensor([0,n],dtype=torch.long,device=device)
    if 'frame_ids' in inputs:inputs['frame_ids']=inputs['frame_ids'].long()
    return inputs

def metric_inputs(observation,target,scene,device,down_factor=8):
    # Construct only after prediction, solely for metric writers.
    full=combine_train(observation,target)
    out={k:v.to(device).float() if torch.is_tensor(v) else v for k,v in full.items()}
    out['abs_pixel_coord']=out['abs_pixel_coord']/down_factor
    out['world_coord']=scene.make_world_coord_torch(out['abs_pixel_coord']*down_factor)
    out['scene']=scene
    return out

@torch.no_grad()
def evaluate_inner(owner,goal=False):
    from src.metrics import compute_metric_mask
    owner.net.eval()
    values={k:[] for k in owner.net.init_test_metrics()}
    for sample in owner.data_loaders['valid']:
        obs=sample['observation'];target=sample['target']
        scene=owner.net.dataset.scenes[sample['identity']['scene_name'][0]]
        inputs=observation_inputs(obs,scene,owner.device,owner.args.down_factor)
        # Future target is never given to prepare_inputs, forward or the sampler.
        if goal:pred=owner.net.forward(inputs)
        else:pred,aux=owner.net.forward(inputs,if_test=True)
        metric=metric_inputs(obs,target,scene,owner.device,owner.args.down_factor)
        seq=metric['seq_list'];mask=compute_metric_mask(seq)
        for name in values:
            if goal:
                v=owner.net.compute_model_metrics(name,pred,metric['abs_pixel_coord'],seq,mask,metric)
            else:
                v=owner.net.compute_model_metrics(metric_name=name,predictions=pred,metric_mask=mask,
                    all_aux_outputs=aux,inputs=metric,obs_length=8)
            values[name].extend(v)
    return {'valid_'+k:sum(v)/len(v) if v else float('nan') for k,v in values.items()}
