"""P2-only cache task execution. Neither plan nor imports construct a model."""
from pathlib import Path
import torch
from src.p2_protocol import CACHE_SCHEMA,digest,data_binding,read_record,validate_payload
from src.joint_dependency_v2_cache import atomic_torch_save,atomic_json_save,sha256_file

def task_plan(registry):
    return [dict(window_id=r['window_id'],role=r['role'],source=r['source'],
        deployment_address=r['source']['physical_split']+'/'+str(r['source']['physical_index']).zfill(6)+'.pt',
        teacher=r['role']=='train') for r in registry.records]

def execute(registry,providers,candidate,root,producer,teacher_fn=None):
    """Same branch for production and CPU fixture; callable receives OBS only."""
    registry.require_provenance()
    if producer.get('classification')!='QUALIFIED' or producer.get('data_binding')!=data_binding(registry):
        raise RuntimeError('P2 cache producer provenance')
    from src.p2_protocol import BAD_BASES
    if producer.get('sha256') in BAD_BASES:raise RuntimeError('KNOWN_PROTOCOL_VIOLATION')
    tasks=task_plan(registry);root=Path(root)
    if root.exists():raise RuntimeError('P2 cache requires a new root; no relabel/overwrite')
    for t in tasks:registry.authorize(t['window_id'],'candidate_build')
    root.mkdir(parents=True)
    ledger=dict(deployment_calls=0,target_reads=0,teacher_calls=0,teacher_writes=0,protected_teacher_calls=0)
    refs={}
    for task in tasks:
        sample=read_record(registry,task['window_id'],'candidate_build',providers)
        r=sample['ref'];obs=sample['observation']
        deployment=candidate(obs,r)
        validate_payload('deployment',deployment)
        ledger['deployment_calls']+=1
        def write(kind,payload,suffix):
            target=root/(task['deployment_address']+suffix)
            atomic_torch_save(dict(window_id=r['window_id'],source_identity=digest(r['source']),
                                   kind=kind,payload=payload),str(target))
            return dict(path=str(target.resolve()),sha256=sha256_file(target))
        refs[r['window_id']]={'deployment':write('deployment',deployment,'')}
        if task['teacher']:
            registry.authorize(r['window_id'],'teacher_build_train')
            target=providers.read(r,'target');ledger['target_reads']+=1
            if teacher_fn is None:
                from src.models.joint_dependency_v2.future_teacher import future_pair_descriptor
                descriptor=future_pair_descriptor(target['future_position_world'],
                    obs['abs_pixel_coord'][-1],deployment['edge_index'])
            else:descriptor=teacher_fn(target,obs,deployment['edge_index'],r)
            ledger['teacher_calls']+=1
            teacher=dict(future_pair_descriptor=descriptor.cpu(),contains_future_supervision=True)
            refs[r['window_id']]['teacher']=write('teacher',teacher,'.teacher')
            ledger['teacher_writes']+=1
    manifest=dict(schema_version=CACHE_SCHEMA,producer_sha256=producer['sha256'],
        data_binding=data_binding(registry),role_manifest_hash=registry.manifest['manifest_hash'],
        K=21,dtype='float32',candidate_order='generate_goal_candidates-v1',coordinate_units='world_m',
        graph=dict(type='radius_ttc',radius=6.,ttc=8.,dt=.4),contains_future_supervision=True,
        teacher_roles=['train'],deployment_contains_future=False,records=refs,ledger=ledger)
    atomic_json_save(manifest,str(root/'manifest.json'))
    return manifest

def build_from_owner(owner):
    from src.p2_protocol import FileProviders
    from src.p2_data import observation_inputs
    from src.models.model import GDTS
    from src.models.model_utils.sampling_2D_map import generate_goal_candidates
    from src.models.interaction_graph import build_interaction_graph
    from src.models.joint_dependency_v2.future_teacher import future_pair_descriptor
    a=owner.args;reg=owner.p2_registry;device=torch.device(a.device)
    model=GDTS(a,device).to(device).eval()
    state=torch.load(owner.checkpoint,map_location='cpu',weights_only=False)['model_state_dict']
    model.goal_module.load_state_dict({k[len('goal_module.'):]:v for k,v in state.items() if k.startswith('goal_module.')},strict=True)
    def scene(r):return model.dataset.scenes[r['source']['scene_family']]
    @torch.no_grad()
    def candidate(obs,r):
        inputs=observation_inputs(obs,scene(r),device,a.down_factor)
        n=inputs['abs_pixel_coord'].shape[1]
        image=inputs['tensor_image'].unsqueeze(0).repeat(n,1,1,1)
        logits=model.goal_module(torch.cat((image,inputs['input_traj_maps']),dim=1))
        goals,prob=generate_goal_candidates(torch.sigmoid(logits[:,-1:]),
            num_candidates=21,device=device,use_ttst=a.use_ttst)
        world=model._map_goals_to_world(goals,inputs['scene'])
        edge,feat,weight=build_interaction_graph(inputs['obs_traj_world'],inputs['scene_index'],
            graph_type=a.graph_type,radius=a.graph_radius,ttc_threshold=a.ttc_threshold,dt=a.trajectory_dt)
        return dict(cache_id=r['window_id'],scene_index=inputs['scene_index'].cpu(),
            scene_ptr=inputs['scene_ptr'].cpu(),frame_ids=inputs['frame_ids'].cpu(),
            edge_index=edge.cpu(),edge_feat=feat.cpu(),edge_weight=weight.cpu(),
            goal_candidates_map=goals.cpu(),goal_candidates_world=world.cpu(),
            candidate_log_prior=prob.float().clamp_min(1e-8).log().cpu(),contains_future_supervision=False)
    def teacher(target,obs,edge,r):
        future=scene(r).make_world_coord_torch(target['abs_pixel_coord']).permute(1,0,2)
        last=scene(r).make_world_coord_torch(obs['abs_pixel_coord'])[-1]
        return future_pair_descriptor(future,last,edge)
    return execute(reg,FileProviders(),candidate,owner.root,reg.manifest['parents']['base'],teacher)
