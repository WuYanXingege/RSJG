#!/usr/bin/env python3
"""Conditional pipeline interface. No automatic training/cache chain."""
import argparse,copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from src.p2_protocol import ROOT,Registry,path,file_hash,digest,sealed,data_binding
from src.p2_grouped import FAMILY,CACHE_SCHEMA,require_provenance
from src.p2_grouped_artifacts import atomic_tensor,write_manifest
from src.p2_observation import atomic_json
from src.p2_grouped_training import qualify_parent,geometry_dataset,seed_all

def dependencies(reg,stage):
    issues=[]
    if stage in ('joint','cache','shared','l1'):
        role='goal' if stage=='joint' else 'base'
        ref=reg.manifest.get('parents',{}).get(role)
        try:qualify_parent(ref,reg,role)
        except (RuntimeError,OSError,KeyError) as e:issues.append(str(e))
    if stage in ('shared','l1'):
        cache=reg.manifest.get('cache')
        if not cache:issues.append('WAITING_FOR_GROUPED_CACHE')
        elif cache.get('schema_version')!=CACHE_SCHEMA or cache.get('data_binding')!=data_binding(reg):issues.append('GROUPED_CACHE_BINDING')
        else:
            if cache.get('producer_sha256')!=(reg.manifest.get('parents',{}).get('base') or {}).get('sha256'):issues.append('CACHE_PARENT_SHA')
            for r in reg.records:
                kinds=['deployment']+(['teacher'] if r['role']=='train' else [])
                if r['role']!='train' and r['artifacts'].get('teacher'):issues.append('PROTECTED_TEACHER')
                for kind in kinds:
                    ref=r['artifacts'].get(kind)
                    if not ref or not path(ref['path']).is_file() or file_hash(path(ref['path']))!=ref['sha256']:issues.append('MISSING_OR_TAMPERED_'+kind.upper())
    if stage=='l1' and not reg.manifest.get('shared_initial_state'):issues.append('WAITING_FOR_SHARED_INITIAL')
    return sorted(set(issues))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['check','prepare-shared'])
    ap.add_argument('--stage',choices=['joint','cache','shared','l1'],default='shared')
    ap.add_argument('--manifest',required=True);ap.add_argument('--manifest-hash',required=True)
    ap.add_argument('--config');ap.add_argument('--output',type=Path);ap.add_argument('--new-manifest',type=Path)
    a=ap.parse_args();reg=Registry(json.loads(path(a.manifest).read_text()),a.manifest_hash)
    if reg.manifest.get('family')!=FAMILY:raise RuntimeError('grouped only')
    issues=dependencies(reg,a.stage)
    if issues:print(json.dumps(dict(status='WAITING',issues=issues)));raise SystemExit(2)
    if a.action=='check':print(json.dumps(dict(status='DEPENDENCIES_VERIFIED_NO_EXECUTION',stage=a.stage)));return
    if a.stage!='shared' or not a.config or not a.output or not a.new_manifest:ap.error('shared requires actual config, new output and new manifest paths')
    if a.output.exists() or a.new_manifest.exists():raise RuntimeError('refuse shared initial overwrite')
    require_provenance(reg,('train',))
    from src.parser import get_parser,compute_term
    from src.models.model import GDTS
    from src.p2_checkpoint import state_hash
    import torch
    cfg=yaml.safe_load(path(a.config).read_text());args=get_parser().parse_args([])
    for k,v in cfg.items():
        if not hasattr(args,k):raise RuntimeError('unknown shared config '+k)
        setattr(args,k,v)
    if args.seed!=3101 or args.jdv2_latent_objective!='strict_no_z' or args.use_scene_latent or args.goal_model_type!='joint_dependency_v2':raise RuntimeError('shared architecture settings')
    args=compute_term(args);args.device='cpu';args.use_cuda=False;args.model_dir=str(a.output.parent)
    args.jdv2_active=True;args.amp_enabled=False;args.amp_dtype='fp32';args.jdv2_goal_objective='mean_energy'
    seed_all(3101,'cpu');model=GDTS(args,'cpu',dataset=geometry_dataset(reg,('train',)))
    base=qualify_parent(reg.manifest['parents']['base'],reg,'base')
    expected={k for k in model.state_dict() if k.startswith(('goal_module.','registrar.','diffnet.','var_sched.'))}
    state=base['model_state_dict']
    if set(state)!=expected:raise RuntimeError('complete baseline state namespace')
    result=model.load_state_dict(state,strict=False)
    if result.unexpected_keys or any(k in expected for k in result.missing_keys):raise RuntimeError('base load mismatch')
    corrector=model.jdv2_corrector
    linear=[x for x in corrector.modules() if isinstance(x,torch.nn.Linear)][-1]
    if not torch.count_nonzero(linear.weight)==0 or not torch.count_nonzero(linear.bias)==0:raise RuntimeError('corrector not zero')
    h=state_hash(model.state_dict());binding=data_binding(reg)
    value=dict(artifact_role='p2_shared_initial',base_sha256=reg.manifest['parents']['base']['sha256'],data_binding=binding,
        initialization_policy='fresh_heads_zero_corrector',optimizer_updates=0,state_sha256=h,model_state_dict=model.state_dict(),seed=3101)
    atomic_tensor(value,a.output);ref=dict(path=str(a.output.resolve()),sha256=file_hash(a.output),base_sha256=value['base_sha256'],data_binding=binding)
    m=copy.deepcopy(reg.manifest);m['shared_initial_state']=ref
    write_manifest(m,reg.records,a.new_manifest);print(json.dumps(dict(status='SHARED_INITIAL_CREATED_NO_UPDATES',reference=ref)))
if __name__=='__main__':main()
