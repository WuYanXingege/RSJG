"""Real-data isolated baseline/optimized microbenchmarks, never heldout access.

The parent launches this file directly with a chosen native source directory;
the worker imports native modules only after replacing sys.path.  Baseline
repeatability and the fixed FP32 tolerance policy are committed to disk before
the first optimized worker.  No production checkpoint is changed.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

BASE_COMMIT = '4969c12adb5675aa654b2e748dec8da0f6e32adf'
SOURCE = Path(__file__).resolve().parents[1]
ASSETS = SOURCE.parent/'RSJG_EBJD_assets/experiments/joint_flow_fourfold_confirmation_20261122'
STAGES = ('density','head','coordinate','dependency')
OPTIMIZATIONS = ('modulation','proposal','schedule','hungarian','geometry','logging','head_cache')
FP32_EPS = 2.0**-23
# Fixed before observing optimized results; never increased on a mismatch.
TOLERANCE_POLICY = dict(output_atol=8*FP32_EPS, output_rtol=8*FP32_EPS,
                       gradient_atol=32*FP32_EPS, gradient_rtol=32*FP32_EPS,
                       update_atol=8*FP32_EPS, update_rtol=8*FP32_EPS,
                       discrete='exact equality, including metric winner indices')


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ids(split,purpose,count):
    """Use the same fixed real-window order in both implementations."""
    ids=split[purpose+'_indices'];n=lambda i:len(split['members'][i]['agent_ids'])
    groups=((lambda x:x==1),(lambda x:2<=x<9),(lambda x:x>=9))
    selected=[]
    for legal in groups:
        eligible=[i for i in ids if legal(n(i))]
        if not eligible:raise ValueError(f'{purpose} missing required real population group')
        selected.append(min(eligible,key=lambda i:(n(i),ids.index(i))))
    for index in ids:
        if len(selected)>=count:break
        if index not in selected:selected.append(index)
    return selected[:count]


def _worker(args):
    sys.path[:]=[str(Path(args.native_source).resolve())]+[p for p in sys.path if p not in (str(SOURCE),str(Path(__file__).parent))]
    import copy
    import random
    import numpy as np
    import torch
    from bcep.contracts import canonical_hash
    from bcep_joint.common import read,seed_all,local_seed
    from bcep_v2.runtime import configure_cuda
    from bcep_v2.bank import Bank
    from bcep_v2.state import optimizer
    from bcep_v2.stages import density_callbacks,head_callbacks,joint_callbacks
    from bcep_v2.provider import spatial_encoding
    from bcep_joint.inference import generate as generate_a,keyed_noise
    from ebjd.unified import gather_candidates
    from ebjd.metrics import per_world_errors,hard_metric_statistics
    from joint_flow_fourfold.inputs import Inputs
    from joint_flow_fourfold.train import evaluate_epoch,RECIPE
    from joint_flow_dependency.inference import generate as generate_b
    from joint_flow_dependency.model import EdgeDependence,delta_nll
    from joint_flow_dependency.sampling import bridge_noise

    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device);torch.set_num_threads(1)
    if device.type=='cuda':
        configure_cuda();torch.cuda.set_per_process_memory_fraction(0.4)
    else:torch.use_deterministic_algorithms(True)
    sync=lambda:torch.cuda.synchronize() if device.type=='cuda' else None
    def telemetry():
        result={}
        if device.type=='cuda':
            for name,query in [('gpu','--query-gpu=name,memory.used,memory.total,utilization.gpu'),('processes','--query-compute-apps=pid,used_memory')]:
                try:result[name]=subprocess.check_output(['nvidia-smi',query,'--format=csv,noheader,nounits'],text=True).strip()
                except (OSError,subprocess.CalledProcessError):result[name]='unavailable'
        return result
    resources_before=telemetry()
    def timed(fn):
        sync();t=time.perf_counter();value=fn();sync();return value,time.perf_counter()-t
    cold=time.perf_counter()
    upstream=read(ASSETS/'zara1/UPSTREAM_FROZEN.json');selected=read(ASSETS/'zara1/B_FROZEN.json')
    assert upstream['joint']['frozen']['epoch']==15 and upstream['joint']['sha256']=='bf85c688ec1fb96bbcea3de1384f54e9f5abd40c91be38294fe3cd11c0eb210f'
    assert selected['rank1_epoch']==6 and selected['rank1_sha256']=='fc7d9cca99769b12f2366eda20396130f547534248c82451e1041d640ff8389d'
    inputs=Inputs(upstream,out/'inputs',device)
    assert inputs.fold=='zara1'
    model,provider=inputs.load_frozen(device)
    dependence=EdgeDependence().to(device)
    assert digest(selected['rank1_checkpoint'])==selected['rank1_sha256']
    dependence.load_state_dict(torch.load(selected['rank1_checkpoint'],map_location='cpu',weights_only=False)['model'],strict=True)
    train16=_ids(inputs.split,'train',16);train4=train16[:4]
    dev3=_ids(inputs.split,'selection_dev',3)
    full_split=inputs.split
    inputs.split=dict(full_split,selection_dev_indices=dev3)
    class SubsetData:
        def __init__(self,data):
            self.data=data;self.split=dict(data.split,selection_dev_indices=dev3);self.io_seconds=0.;self.read_calls=0
        def item(self,*a,**kw):
            begin=time.perf_counter();v=self.data.item(*a,**kw);self.io_seconds+=time.perf_counter()-begin;self.read_calls+=1;return v
        def batch(self,*a,**kw):
            begin=time.perf_counter();v=self.data.batch(*a,**kw);self.io_seconds+=time.perf_counter()-begin;self.read_calls+=1;return v
    data=SubsetData(inputs.data);inputs.data=data
    base_folder=Path(upstream['joint']['checkpoint']).parent.parent/'base_cache'
    base_train=[];base_dev=[]
    for purpose,seeds,target in [('train',[local_seed('bcep-v2-train-view',inputs.full_config['provider_seed'],v) for v in range(2)],base_train),
                                  ('selection_dev',range(2035,2040),base_dev)]:
        for seed in seeds:
            folder=base_folder/f'{purpose}_{seed}';target.append(Bank(folder,read(folder/'IDENTITY.json'),full_split))
    latent_root=ASSETS/'zara1/dependency/latent_cache'
    latent_identity=read(latent_root/'IDENTITY.json')
    assert latent_identity==dict(inputs.identity,schema='frozen-flow-synchronous-endpoint-latents-v1',precision='fp32')
    cache={}
    for purpose,wanted in [('train',train16),('selection_dev',dev3)]:
        manifest=read(latent_root/f'{purpose}_MANIFEST.json');records=[]
        for index in wanted:
            pos=full_split[purpose+'_indices'].index(index);entry=manifest[pos];path=latent_root/entry['file']
            assert digest(path)==entry['sha256']
            record=torch.load(path,map_location='cpu',weights_only=False)
            assert record['index']==index and record['identity']==canonical_hash(latent_identity)
            records.append(record)
        cache[purpose]=records
    frozen={name:copy.deepcopy(net.state_dict()) for name,net in [('model',model),('provider',provider),('dependency',dependence)]}
    cold_seconds=time.perf_counter()-cold
    fixed_io_start=data.io_seconds
    for purpose,indices in [('train',train16),('selection_dev',dev3)]:
        for index in indices:data.batch([index],purpose,truth=True)
    initial_io_seconds=data.io_seconds-fixed_io_start
    networks=dict(model=model,provider=provider,dependency=dependence)
    artifact={};result={}
    def reset(stage):
        for name,net in networks.items():net.load_state_dict(frozen[name],strict=True);net.eval().requires_grad_(False);net.zero_grad(set_to_none=True)
        net=provider.base if stage=='density' else provider.head if stage=='head' else model if stage=='coordinate' else dependence
        net.train().requires_grad_(True)
        return net
    def cpu(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
        if isinstance(value,dict):return {str(k):cpu(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [cpu(v) for v in value]
        if isinstance(value,(int,float,bool,str)) or value is None:return value
        raise TypeError(type(value))
    def vector(net,gradient=False):
        values=[];names=[]
        for name,p in net.named_parameters():
            if not p.requires_grad:continue
            value=p.grad if gradient else p
            if value is None:
                values.append(torch.zeros_like(p).flatten());names.append(name+':NO_GRAD')
            else:values.append(value.detach().flatten());names.append(name)
        return torch.cat(values).cpu(),names
    def rng():
        return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                    python=random.getstate(),numpy=np.random.get_state())
    def restore_rng(state):
        torch.set_rng_state(state['cpu']);random.setstate(state['python']);np.random.set_state(state['numpy'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(state['cuda'])
    def callbacks(stage):
        if stage=='density':return density_callbacks(provider,data,device)
        if stage=='head':return head_callbacks(provider,data,device,inputs.geometry,base_train,base_dev)
        if stage=='coordinate':return joint_callbacks(model,data,device,inputs.config,inputs.banks['train'],inputs.banks['selection_dev'])
        mapping={r['index']:r for r in cache['train']}
        def backward(ids,order,views):
            del order,views
            total=0.
            for index in ids:
                record=mapping[index]
                R=dependence(**{k:v.to(device) for k,v in record['inputs'].items()})['R']
                loss=delta_nll(R,record['latent_truth'].to(device));(loss/len(ids)).backward();total+=float(loss.detach())
            return dict(delta_nll_nat_per_person=total/len(ids),delta_nll_window_sum=total,equally_weighted_windows=len(ids))
        @torch.no_grad()
        def validate():
            inputs.output=out/'dependency_native_validation'
            metrics=evaluate_epoch(inputs,model,provider,dependence,cache['selection_dev'],device,epoch=1)
            validate.metrics=metrics
            return metrics['all']['JADE']
        return backward,validate
    def init_update(stage):
        net=reset(stage);seed_all(3201)
        cfg=inputs.full_config
        lr=cfg['provider']['candidate_head']['lr'] if stage=='head' else 1e-4
        opt=optimizer(net,lr,1e-4)
        return net,opt,torch.Generator().manual_seed(3201),torch.Generator().manual_seed(3202)
    def single_update(net,opt,order,views,backward,ids,clip):
        opt.zero_grad(set_to_none=True);net.train()
        metrics=backward(ids,order,views)
        torch.nn.utils.clip_grad_norm_(net.parameters(),clip)
        opt.step()
        return metrics
    def witness(stage):
        # Actual dev outputs from unchanged native entrances; no new prediction rule.
        rows=[]
        with torch.no_grad():
            for index in dev3:
                host=data.batch([index],'selection_dev',truth=False);batch=host.to(device)
                if stage=='density':
                    h,_=spatial_encoding(provider.base,batch.observed[0],batch.semantic_maps[0],batch.valid[0]);rows.append(dict(context=h));continue
                if stage=='head':
                    h,features=spatial_encoding(provider.base,batch.observed[0],batch.semantic_maps[0],batch.valid[0])
                    goals=base_dev[0].batch([index],host)[0].to(device)
                    refined,shift=provider.head(provider.base,batch.observed[0],batch.semantic_maps[0],batch.valid[0],goals,h,features,inputs.geometry)
                    rows.append(dict(refined=refined,shift=shift));continue
                if stage=='coordinate':
                    goals=inputs.banks['selection_dev'].batch([index],host).to(device)
                    _,path,pi,proof=generate_a(model,batch.observed,batch.semantic_maps,batch.valid,batch.scene_ids,batch.agent_ids,goals,2035,mode='H')
                    noise=gather_candidates(keyed_noise(batch.scene_ids,batch.agent_ids,batch.valid,2035).to(device),pi)
                    extra=dict(permutation=pi,noise=noise,proof=proof)
                else:
                    path,endpoints,detail=generate_b(model,provider,dependence,batch,seed=2036)
                    noise=bridge_noise(batch.scene_ids[0],batch.agent_ids[0],endpoints['raw_indices'].cpu().tolist(),seed=2036).to(device)
                    assert torch.equal(noise,endpoints['noise'])
                    extra={key:endpoints[key] for key in ('raw_indices','noise','raw','corrected','epsilon','xi')}
                truth=data.batch([index],'selection_dev',truth=True).future.to(device)
                ade,fde=per_world_errors(path,truth,batch.valid);n=batch.valid.sum(1)
                winners=dict(minade=ade.argmin(-1),minfde=fde.argmin(-1),
                    jade=(ade.sum(1)/n[:,None]).argmin(-1),jfde=(fde.sum(1)/n[:,None]).argmin(-1))
                stats=hard_metric_statistics(path,truth,batch.valid)
                rows.append(dict(path=path,**extra,winners=winners,metrics=stats.__dict__))
        return cpu(rows)

    chain_start=time.perf_counter()
    for stage in STAGES:
        ids=train4 if stage in ('density','head') else train16
        clip=1. if stage in ('density','head') else 5.
        net,opt,order,views=init_update(stage);backward,validate=callbacks(stage)
        initial_trainable=copy.deepcopy(net.state_dict())
        single_update(net,opt,order,views,backward,ids,clip);net.eval()
        _,cold_validation_seconds=timed(validate)
        cache_after_cold=cpu(getattr(validate,'context_cache_info',None))
        # Reuse the exact frozen base and validation callback across epochs.
        # Only the trainable head is reset; its frozen-base cache remains valid.
        net.load_state_dict(initial_trainable);seed_all(3201)
        lr=inputs.full_config['provider']['candidate_head']['lr'] if stage=='head' else 1e-4
        opt=optimizer(net,lr,1e-4);order=torch.Generator().manual_seed(3201);views=torch.Generator().manual_seed(3202)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        io_before=data.io_seconds
        update_metrics,update_seconds=timed(lambda:single_update(net,opt,order,views,backward,ids,clip))
        # Audit transfers are outside the production update timing.
        gradient,names=vector(net,True)
        updated,parameter_names=vector(net)
        net.eval();validation,validation_seconds=timed(validate)
        measured_read_seconds=data.io_seconds-io_before
        # Validation may include wall-time diagnostics, which are excluded from equality.
        val_metrics=cpu(validate.metrics)
        def strip_times(d):
            if isinstance(d,dict):return {k:strip_times(v) for k,v in d.items() if k not in ('seconds',)}
            if isinstance(d,list):return [strip_times(v) for v in d]
            return d
        val_metrics=strip_times(val_metrics)
        outputs=witness(stage)
        # Reproduce the next real update from an atomic complete-state snapshot.
        state=dict(model=copy.deepcopy(net.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=rng(),
            order_rng=order.get_state(),view_rng=views.get_state(),completed_epoch=0,update=1,cursor=len(ids),
            data_order=list(ids),stage=stage,benchmark_only=True)
        path=out/f'{stage}_recovery.pt';temporary=path.with_suffix('.tmp.pt');torch.save(state,temporary);temporary.replace(path)
        single_update(net,opt,order,views,backward,ids,clip);continuous,_=vector(net)
        saved=torch.load(path,map_location=device,weights_only=False)
        net.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);restore_rng(state['rng'])
        order.set_state(state['order_rng']);views.set_state(state['view_rng'])
        single_update(net,opt,order,views,backward,ids,clip);recovered,_=vector(net)
        assert torch.equal(continuous,recovered),'same-implementation update recovery differs'
        artifact[stage]=dict(training_metrics=cpu(update_metrics),gradient=gradient,gradient_names=names,
            updated_parameters=updated,parameter_names=parameter_names,validation=validation,
            validation_metrics=val_metrics,outputs=outputs,recovery_exact=True)
        result[stage]=dict(logical_windows=len(ids),validation_windows=len(dev3),warmups=1,
            update_seconds=update_seconds,validation_seconds=validation_seconds,
            cold_validation_seconds=cold_validation_seconds,
            context_cache_after_cold=cache_after_cold,context_cache_after_warm=cpu(getattr(validate,'context_cache_info',None)),
            measured_read_seconds=measured_read_seconds,
            io_scope='native update and validation reads only; audit/warmup reads excluded',
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if device.type=='cuda' else None,
            recovery_exact=True,formal_training_epochs=0)
        print(stage,result[stage],flush=True)
    four_stage_fixture_wall=time.perf_counter()-chain_start
    # Full observed->Flow/dependence->correction->bridge, independently warmed.
    reset('dependency');dependence.eval().requires_grad_(False)
    def chain():
        values=[]
        with torch.no_grad():
            for index in dev3:
                batch=data.batch([index],'selection_dev',truth=False).to(device)
                path,endpoints,_=generate_b(model,provider,dependence,batch,seed=2036)
                values.append(dict(path=path,raw_indices=endpoints['raw_indices'],noise=endpoints['noise']))
        return cpu(values)
    chain();chain_values,chain_seconds=timed(chain);artifact['observed_to_B']=chain_values
    torch.save(artifact,out/'WITNESSES.pt')
    # Only anonymous positions/counts are made public in the scalar receipt.
    anonymous={purpose:[dict(ordinal=full_split[purpose+'_indices'].index(i)+1,n=len(full_split['members'][i]['agent_ids'])) for i in ids]
        for purpose,ids in [('train',train16),('selection_dev',dev3)]}
    resources_after=telemetry()
    write(out/'RESULT.json',dict(status='COMPLETE',device=str(device),dtype='float32',K=20,M=256,DDIM=20,
        train_seed=3201,inference_seed=2036,native_validation_seed=2035,selection=anonymous,
        upstream_manifest_sha256=digest(ASSETS/'zara1/UPSTREAM_FROZEN.json'),B_manifest_sha256=digest(ASSETS/'zara1/B_FROZEN.json'),
        native_source=str(Path(args.native_source).resolve()),stages=result,
        cold_start_load_and_verify_seconds=cold_seconds,initial_data_read_seconds=initial_io_seconds,
        cold_io_scope='first reads in this worker; OS page cache not flushed',
        cached_assets='read existing native base/refined banks and frozen context/latent cache; no full candidate or latent cache rebuild',
        resource_telemetry=dict(before=resources_before,after=resources_after,mode='shared GPU with original training; paired timings are not isolated-GPU claims'),
        disabled_optimizations=sorted(filter(None,os.environ.get('RSJG_DISABLE_OPTIMIZATIONS','').split(','))),
        four_stage_fixture_wall_seconds=four_stage_fixture_wall,
        four_stage_fixture_includes='warmup, native updates/validation, witnesses and recovery checks; not a production full training run',
        observed_to_B_seconds=chain_seconds,observed_to_B_windows=len(dev3),
        witnesses_sha256=digest(out/'WITNESSES.pt'),test_access=0,production_checkpoint_updates=0))


def _flatten(value,prefix=''):
    import torch
    if isinstance(value,dict):
        for key,item in value.items():yield from _flatten(item,prefix+'/'+str(key))
    elif isinstance(value,(list,tuple)):
        for i,item in enumerate(value):yield from _flatten(item,prefix+'/'+str(i))
    elif isinstance(value,torch.Tensor):yield prefix,value
    elif isinstance(value,(bool,int,float)):yield prefix,torch.tensor(value,dtype=torch.float64 if isinstance(value,float) else None)
    elif value is None or isinstance(value,str):yield prefix,value
    else:raise TypeError(type(value))


def compare(a_path,b_path,tolerances):
    import torch
    a=dict(_flatten(torch.load(a_path,map_location='cpu',weights_only=False)))
    b=dict(_flatten(torch.load(b_path,map_location='cpu',weights_only=False)))
    assert set(a)==set(b),'witness structure changed'
    failures=[];maxima={};checks=0
    for name,x in a.items():
        y=b[name];checks+=1
        if not isinstance(x,torch.Tensor):
            if x!=y:failures.append(dict(field=name,reason='non-tensor identity differs'))
            continue
        if x.shape!=y.shape or x.dtype!=y.dtype:
            failures.append(dict(field=name,reason='shape or dtype differs'));continue
        exact=(not x.is_floating_point()) or any(key in name for key in ('/raw_indices','/permutation','/noise','/epsilon','/xi','/winners/'))
        if exact:
            if not torch.equal(x,y):failures.append(dict(field=name,reason='strict identity/index/noise differs'))
            continue
        category='gradient' if '/gradient' in name else 'update' if '/updated_parameters' in name else 'output'
        diff=float((x.double()-y.double()).abs().max()) if x.numel() else 0.
        maxima[category]=max(maxima.get(category,0.),diff)
        atol=tolerances[category+'_atol'];rtol=tolerances[category+'_rtol']
        if not torch.allclose(x,y,atol=atol,rtol=rtol,equal_nan=False):
            failures.append(dict(field=name,reason='fixed FP32 tolerance exceeded',max_abs=diff,atol=atol,rtol=rtol))
    return dict(passed=not failures,checks=checks,max_abs_by_category=maxima,failures=failures)


def _launch(args,source,folder,disabled=()):
    if (folder/'RESULT.json').exists():
        saved=json.loads((folder/'RESULT.json').read_text())
        assert saved['status']=='COMPLETE' and saved['device']==args.device
        assert saved['native_source']==str(Path(source).resolve())
        assert saved['disabled_optimizations']==sorted(disabled)
        assert saved['witnesses_sha256']==digest(folder/'WITNESSES.pt')
        return saved
    folder.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             RSJG_DISABLE_OPTIMIZATIONS=','.join(sorted(disabled)))
    with (folder/'worker.log').open('w') as log:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker','--native-source',str(source),
            '--output',str(folder),'--device',args.device],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    return json.loads((folder/'RESULT.json').read_text())


def implicated_optimization(failures,disabled):
    """Predeclared conservative order; retire one named change, then recheck."""
    fields=' '.join(row['field'] for row in failures)
    if '/permutation' in fields:order=['hungarian','modulation','proposal','schedule','geometry','logging','head_cache']
    elif '/density/' in fields:order=['logging','head_cache','modulation','proposal','schedule','geometry','hungarian']
    elif '/head/' in fields:order=['head_cache','logging','modulation','proposal','schedule','geometry','hungarian']
    elif '/gradient' in fields:order=['modulation','proposal','geometry','logging','hungarian','schedule','head_cache']
    elif '/training_metrics' in fields:order=['logging','geometry','modulation','proposal','schedule','hungarian','head_cache']
    else:order=['proposal','schedule','modulation','head_cache','hungarian','geometry','logging']
    return next((name for name in order if name not in disabled),None)


def performance_rejections(observations,disabled):
    """Require >=2% median improvement and at least 2/3 faster paired trials.

    These grouped associations are conservative admission evidence, not causal
    estimates of individual kernel contributions.
    """
    lookup={(r['variant'],r['trial']):r for r in observations}
    checks=[(('head_cache',),lambda r:r['stages']['head']['validation_seconds'],'head_warm_validation'),
            (('logging',),lambda r:r['stages']['density']['update_seconds']+r['stages']['density']['validation_seconds'],'density_update_plus_validation'),
            (('proposal','schedule'),lambda r:r['observed_to_B_seconds'],'observed_to_B'),
            (('geometry','hungarian'),lambda r:r['stages']['coordinate']['update_seconds'],'coordinate_update'),
            (('modulation',),lambda r:r['stages']['coordinate']['update_seconds']+r['stages']['coordinate']['validation_seconds'],'coordinate_update_plus_validation')]
    rejected=[]
    for associated,value,label in checks:
        active=[name for name in associated if name not in disabled]
        if not active:continue
        base=[value(lookup['baseline',i]) for i in range(3)]
        new=[value(lookup['optimized',i]) for i in range(3)]
        ratio=statistics.median(base)/statistics.median(new)
        wins=sum(a>b for a,b in zip(base,new))
        if ratio<1.02 or wins<2:
            rejected.append(dict(optimizations=active,evidence=label,median_speed_ratio=ratio,paired_faster_trials=wins,
                                 reason='no clear timing improvement under the fixed shared-resource admission rule'))
    return rejected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--baseline-source',type=Path)
    parser.add_argument('--native-source',type=Path)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker:return _worker(args)
    assert args.baseline_source is not None
    baseline=args.baseline_source.resolve();optimized=SOURCE
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    # Source identity excludes documentation-only commits; baseline must retain the fixed native code.
    packages=('bcep','bcep_v2','bcep_joint','bcep_formal','ebjd','joint_flow_dependency','joint_flow_fourfold')
    sources={name:{str(p.relative_to(root)):digest(p) for package in packages for p in sorted((root/package).rglob('*.py'))}
             for name,root in [('baseline',baseline),('optimized',optimized)]}
    for relative,sha in sources['baseline'].items():
        expected=subprocess.check_output(['git','show',BASE_COMMIT+':'+relative],cwd=baseline)
        assert hashlib.sha256(expected).hexdigest()==sha,'baseline native source is not fixed commit'
    config=dict(base_commit=BASE_COMMIT,sources=sources,benchmark_source_sha256=digest(__file__),
        device=args.device,FP32=True,repeat_pairs=3,baseline_calibration_repeats=2,tolerance_policy=TOLERANCE_POLICY,
        resource_policy='user authorized parallel GPU; at most one auxiliary benchmark worker, 40% allocator cap; record existing training contention',
        optimization_admission='correctness first, named one-at-a-time conservative rollback; then >=2% median improvement and >=2/3 faster pairs for the associated component',
        production_full_training_repeated=False,test_access=0)
    path=output/'CONFIG.json'
    if path.exists():assert json.loads(path.read_text())==config,'bound benchmark source/config changed'
    else:write(path,config)
    _launch(args,baseline,output/'calibration_0');_launch(args,baseline,output/'calibration_1')
    calibration=compare(output/'calibration_0/WITNESSES.pt',output/'calibration_1/WITNESSES.pt',TOLERANCE_POLICY)
    # Calibration is a prerequisite, never used to relax tolerance after an optimized failure.
    assert calibration['passed'],'baseline itself exceeds the predeclared FP32 budget'
    lock=dict(policy=TOLERANCE_POLICY,baseline_repeat=calibration,
              baseline_witness_sha256=[digest(output/f'calibration_{i}/WITNESSES.pt') for i in (0,1)])
    if (output/'TOLERANCE_LOCK.json').exists():
        saved=json.loads((output/'TOLERANCE_LOCK.json').read_text())
        assert {k:v for k,v in saved.items() if k!='locked_before_optimized_unix'}==lock
    else:write(output/'TOLERANCE_LOCK.json',dict(lock,locked_before_optimized_unix=time.time()))
    disabled=set(filter(None,os.environ.get('RSJG_DISABLE_OPTIMIZATIONS','').split(',')))
    assert disabled.issubset(OPTIMIZATIONS)
    decisions=[];accepted=False
    for policy_index in range(len(OPTIMIZATIONS)+2):
        folder=output/f'policy_{policy_index:02d}';folder.mkdir(exist_ok=True)
        policy=dict(disabled_optimizations=sorted(disabled),decision_history=decisions)
        if (folder/'POLICY.json').exists():assert json.loads((folder/'POLICY.json').read_text())==policy
        else:write(folder/'POLICY.json',policy)
        observations=[];comparisons=[];failed=False
        for trial in range(3):
            variants=[('baseline',baseline),('optimized',optimized)]
            if trial%2:variants.reverse()
            for name,source in variants:
                result=_launch(args,source,folder/f'{name}_{trial}',disabled if name=='optimized' else ())
                observations.append(dict(variant=name,trial=trial,**result))
            comparison=compare(folder/f'baseline_{trial}/WITNESSES.pt',folder/f'optimized_{trial}/WITNESSES.pt',TOLERANCE_POLICY)
            comparisons.append(dict(trial=trial,**comparison))
            write(folder/'CORRECTNESS.json',dict(passed=all(v['passed'] for v in comparisons),pairs=comparisons))
            if not comparison['passed']:
                switch=implicated_optimization(comparison['failures'],disabled)
                if switch is None:
                    write(output/'REJECTED.json',dict(status='ALL_OPTIMIZATIONS_DISABLED_BUT_NATIVE_EQUIVALENCE_FAILED',failures=comparison['failures']))
                    raise RuntimeError('native mismatch persists with every optimization off')
                decision=dict(policy=policy_index,reason='fixed correctness budget failed',disabled=[switch],failures=comparison['failures'])
                decisions.append(decision);disabled.add(switch);failed=True;break
        if failed:continue
        rejected=performance_rejections(observations,disabled)
        write(folder/'PERFORMANCE_ADMISSION.json',dict(rejected=rejected,threshold_median_speed_ratio=1.02,minimum_faster_pairs=2))
        if rejected:
            newly=sorted({name for row in rejected for name in row['optimizations']})
            decisions.append(dict(policy=policy_index,reason='no clear timing benefit',disabled=newly,evidence=rejected))
            disabled.update(newly);continue
        accepted=True;break
    assert accepted,'optimization resolution exhausted without a final verified configuration'
    write(output/'CORRECTNESS.json',dict(passed=True,pairs=comparisons,final_policy=policy_index))
    rows=[]
    for stage in STAGES:
        for part in ('update_seconds','validation_seconds'):
            values={name:[r['stages'][stage][part] for r in observations if r['variant']==name] for name in ('baseline','optimized')}
            base=statistics.median(values['baseline']);new=statistics.median(values['optimized'])
            rows.append(dict(stage=stage,component=part,baseline_median_seconds=base,optimized_median_seconds=new,
                speed_ratio=base/new,paired_trials=3,clearly_estimated_full_training=False))
    for part in ('cold_start_load_and_verify_seconds','initial_data_read_seconds','four_stage_fixture_wall_seconds','observed_to_B_seconds'):
        values={name:[r[part] for r in observations if r['variant']==name] for name in ('baseline','optimized')}
        base=statistics.median(values['baseline']);new=statistics.median(values['optimized'])
        rows.append(dict(stage='pipeline',component=part,baseline_median_seconds=base,optimized_median_seconds=new,
            speed_ratio=base/new,paired_trials=3,clearly_estimated_full_training=False))
    with (output/'TIMINGS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(output/'RESULT.json',dict(status='COMPLETE',correctness_passed=True,real_fold='zara1',
        paired_repeats=3,timings=rows,observations=observations,baseline_repeat=calibration,
        disabled_optimizations=sorted(disabled),optimization_resolution=decisions,final_policy=policy_index,
        full_old_training_runs_measured=False,student_full_fold_seconds='reported by formal head controller separately',
        interpretation='native four-stage representative logical updates/three-window dev, and observed-to-B chain; no full-network training speedup inferred from different denominators'))


if __name__=='__main__':main()
