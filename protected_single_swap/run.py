"""Frozen single-swap search, global endpoint seal, then development scoring."""
import argparse,fcntl,os,subprocess,sys,time,traceback
import numpy as np
import torch
from hotel_endpoint_diagnostic.run import prediction_guard,validate_shard as validate_cache
from joint_flow_fourfold.common import immutable
from joint_flow_fourfold.inputs import state_hash
from joint_marginal_residual.assets import load_teacher
from joint_marginal_residual.run import protect
from joint_flow_dependency.inference import encode_observed
from .common import *
from .math import rank_positive_swaps,coverage_cost,directed_distance
from .head import directed_distance_batch
from .native_graph import NativeHeadGraph


def status(value,**kwargs):
    obj=dict(status=value,unix=time.time(),**kwargs);write(RUN/'STATUS.json',obj);print(obj,flush=True)


def prepare():
    RUN.mkdir(parents=True,exist_ok=True);protect()
    old=read(CACHE/'RUN_CONFIG.json');seal=read(CACHE/'PREDICTIONS_FROZEN.json')
    assert seal['config_sha256']==digest(CACHE/'RUN_CONFIG.json') and len(seal['rows'])==688
    with prediction_guard():
        inputs,model,provider,dep,identity=load_teacher('hotel','cpu')
        assert identity==old['identity'] and identity['teacher_epoch']==19
        ids=inputs.split['selection_dev_indices'];assert len(ids)==688
        for index,row in zip(ids,seal['rows']):
            assert row['index']==index and row['member_sha256']==canonical_hash(inputs.split['members'][index])
            assert row['sha256']==digest(CACHE/row['file'])
            validate_cache(CACHE/row['file'],row['n'])
    assert sum(r['n'] for r in seal['rows'])==5203
    config=dict(base_commit=BASE,fold='hotel',purpose='selection_dev',seed=2035,M=256,K=20,
        windows=688,agents=5203,identity=identity,reference=REFERENCE,source_sha256=sources(),
        cache_manifest_sha256=digest(CACHE/'PREDICTIONS_FROZEN.json'),cache_config_sha256=digest(CACHE/'RUN_CONFIG.json'),
        cache_source_commit=BASE,raw_reused_windows=688,raw_generated_windows=0,
        budget_m=BUDGET,score_absolute_tolerance_m=1e-5,native_replay_tolerance=dict(atol=1e-6,rtol=1e-6),
        fp64_distance_screen_roundoff_m=1e-12,head_chunk=16,
        precision='CPU FP64 exact coverage and directed final distance; deterministic frozen CUDA FP32 CandidateHead',
        proposal='enumerate all 20*236 swaps; strictly gain>0; decreasing FP64 gain, outgoing original ID, incoming original ID; incoming occupies outgoing slot',
        protection='max_old min_new mean_valid_agent Euclidean displacement; final unchanged native CandidateHead; h<=0.005 strictly',
        acceleration='CUDA Graph replay of separate native-shaped CandidateHead tensor bodies, validation outside capture; no algebraic or batch-GEMM changes; every earlier positive-gain candidate evaluated',
        gates=dict(endpoint_minFDE_improvement_m=.020,all_windows_protected=True,
                   conditional_DDIM_steps=20,conditional_eta=0.,further_folds_only_if_HOTEL_JADE_and_minADE_not_higher=True),
        native_disabled_optimizations=old['native_disabled_optimizations'],training_updates=0,test_access=0,
        motion=old['motion'],difference='new minus native B',
        resources='existing auxiliary_gpu.lock; one worker; 40 percent allocator cap; start only with >=6GiB free; two CPU threads',
        stop='If all-agent minFDE reduction <0.020 m, end after endpoint scoring with DDIM0; no budget relaxation or parameter search')
    immutable(RUN/'RUN_CONFIG.json',config);return config


def validate_new(path,n):
    with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    for k,shape in [('new_end',(20,n,2)),('old_end',(20,n,2)),('observed',(n,8,2))]:
        assert a[k].shape==shape and a[k].dtype==np.float32 and np.isfinite(a[k]).all()
    assert a['valid'].shape==(n,) and a['valid'].all()
    for k in ('chosen_ids','native_ids'):
        assert a[k].shape==(20,) and len(np.unique(a[k]))==20 and a[k].min()>=0 and a[k].max()<256
    assert np.count_nonzero(a['chosen_ids']!=a['native_ids'])<=1
    assert directed_distance(a['old_end'],a['new_end'],a['valid'])<=BUDGET
    return a


@torch.no_grad()
def search_window(provider,batch,context,features,a,config,graph):
    start=time.monotonic();rank=rank_positive_swaps(a['raw'],a['native_ids'],a['valid'])
    assert rank['total_swaps']==4720 and rank['p0']==coverage_cost(a['raw'],a['native_ids'],a['valid'])
    raw=torch.from_numpy(a['raw']).to(batch.observed.device)
    native_start=graph.native_calls
    native=graph.native(raw,a['native_ids']).cpu().numpy()
    np.testing.assert_allclose(native,a['native_post'],atol=1e-6,rtol=1e-6)
    old=a['native_post'];ids=a['native_ids'].copy();new=old.copy();chosen=None
    replay=float(np.max(np.abs(native-old)));parity=0.;screened=0;native_rejections=0
    candidates=rank['swaps'];chunk=config['head_chunk']
    # Each graph slot runs the same native tensor body and matrix shapes.
    # Validate exact replay at native, first and last set in every window.
    check_ids=[ids.copy()]
    for row in candidates[:1]+candidates[-1:]:
        value=ids.copy();value[row['slot']]=row['in_id'];check_ids.append(value)
    approx=graph.evaluate(raw,np.stack(check_ids)).cpu().numpy()
    for p,v in zip(approx,check_ids):
        exact=graph.native(raw,v).cpu().numpy();drift=float(np.linalg.norm(p.astype(np.float64)-exact.astype(np.float64),axis=-1).max())
        parity=max(parity,drift);assert np.array_equal(p,exact),'graph/native head arithmetic differs'
    for start_rank in range(0,len(candidates),chunk):
        rows=candidates[start_rank:start_rank+chunk]
        values=np.tile(ids,(len(rows),1))
        for j,row in enumerate(rows):values[j,row['slot']]=row['in_id']
        proposed=graph.evaluate(raw,values)
        fast_h=directed_distance_batch(torch.from_numpy(old),proposed,batch.valid[0]).cpu().numpy()
        screened+=len(rows)
        for j in np.flatnonzero(fast_h<=BUDGET+config['fp64_distance_screen_roundoff_m']):
            # Original single-set head is authoritative; equality at .005 is accepted.
            exact=graph.native(raw,values[j]).cpu().numpy();p=proposed[j].cpu().numpy()
            drift=float(np.linalg.norm(p.astype(np.float64)-exact.astype(np.float64),axis=-1).max())
            parity=max(parity,drift);assert np.array_equal(p,exact)
            h=directed_distance(old,exact,a['valid'])
            if h<=BUDGET:
                chosen=dict(rows[j],rank=start_rank+int(j)+1);ids=values[j].copy();new=exact;break
            native_rejections+=1
        if chosen is not None:break
    pnew=coverage_cost(a['raw'],ids,a['valid']);h=directed_distance(old,new,a['valid'])
    assert h<=BUDGET
    if chosen is not None:
        assert pnew==chosen['p'] and rank['p0']-pnew>0
        assert np.count_nonzero(ids!=a['native_ids'])==1
    else:assert np.array_equal(new,old) and pnew==rank['p0']
    stats=dict(p0=rank['p0'],pnew=pnew,gain=rank['p0']-pnew,positive_swaps=len(candidates),
        head_evaluated=screened,accepted_rank=chosen['rank'] if chosen else 0,replaced=chosen is not None,
        fallback_reason=None if chosen else ('no_positive_gain' if not candidates else 'no_feasible'),h_m=h,
        seconds=time.monotonic()-start,native_endpoint_replay_max_abs_m=replay,
        head_parity_max_endpoint_norm_m=parity,native_head_calls=graph.native_calls-native_start,
        graph_head_sets=screened+len(check_ids),native_protection_rejections=native_rejections,
        out_id=chosen['out_id'] if chosen else None,in_id=chosen['in_id'] if chosen else None)
    arrays=dict(new_end=new,old_end=old,chosen_ids=ids,native_ids=a['native_ids'],observed=a['observed'],valid=a['valid'])
    return arrays,stats


@torch.no_grad()
def export():
    from bcep_v2.runtime import configure_cuda
    config=read(RUN/'RUN_CONFIG.json');assert sources()==config['source_sha256'];protect()
    torch.set_num_threads(2);configure_cuda();torch.cuda.set_per_process_memory_fraction(.4)
    os.environ['RSJG_DISABLE_OPTIMIZATIONS']=config['native_disabled_optimizations']
    oldseal=read(CACHE/'PREDICTIONS_FROZEN.json');assert digest(CACHE/'PREDICTIONS_FROZEN.json')==config['cache_manifest_sha256']
    (RUN/'predictions').mkdir(exist_ok=True);rows=[];start=time.time()
    with prediction_guard():
        inputs,model,provider,dep,identity=load_teacher('hotel','cuda');assert identity==config['identity']
        modules=dict(backbone=model,provider=provider,dependence=dep);before={k:state_hash(m) for k,m in modules.items()}
        def no_ddim(*args,**kwargs):raise RuntimeError('DDIM prohibited until endpoint gate passes')
        model.bridge.forward=no_ddim;model.bridge.denoiser.forward=no_ddim
        graphs={}
        for pos,original in enumerate(oldseal['rows']):
            path=RUN/'predictions'/f"{original['window']}.npz";side=path.with_suffix('.json')
            expected={k:original[k] for k in ('window','index','n','member_sha256')}
            expected.update(file=str(path.relative_to(RUN)),config_sha256=digest(RUN/'RUN_CONFIG.json'),cache_sha256=original['sha256'])
            if side.exists():
                row=read(side);assert {k:row[k] for k in expected}==expected and digest(path)==row['sha256']
                a=validate_new(path,original['n']);assert directed_distance(a['old_end'],a['new_end'],a['valid'])==row['stats']['h_m']
            else:
                assert digest(CACHE/original['file'])==original['sha256'];a=validate_cache(CACHE/original['file'],original['n'])
                host=inputs.data.batch([original['index']],'selection_dev',truth=False)
                assert not host.future.count_nonzero() and np.array_equal(host.observed[0].numpy(),a['observed'])
                assert np.array_equal(host.valid[0].numpy(),a['valid'])
                batch=host.to('cuda');context,features=encode_observed(provider,batch)
                n=original['n']
                if n not in graphs:graphs[n]=NativeHeadGraph(provider,batch,context,features,chunk=config['head_chunk'])
                else:graphs[n].update(batch,context,features)
                values,stats=search_window(provider,batch,context,features,a,config,graphs[n])
                if path.exists():
                    prior=validate_new(path,original['n']);assert all(np.array_equal(prior[k],v) for k,v in values.items())
                else:
                    tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**values);tmp.replace(path)
                validate_new(path,original['n']);row=dict(expected,sha256=digest(path),stats=stats);write(side,row)
            rows.append(row)
            if (pos+1)%20==0 or pos==687:status('SEARCHING_OBSERVED_ONLY',windows=pos+1,total=688,
                replacements=sum(r['stats']['replaced'] for r in rows),elapsed_seconds=time.time()-start)
        after={k:state_hash(m) for k,m in modules.items()};assert before==after
        for row in rows:assert digest(RUN/row['file'])==row['sha256'];validate_new(RUN/row['file'],row['n'])
        immutable(RUN/'PREDICTIONS_FROZEN.json',dict(rows=rows,seal_unix=time.time(),future_reads=0,
            config_sha256=digest(RUN/'RUN_CONFIG.json'),cache_manifest_sha256=config['cache_manifest_sha256'],
            weights_unchanged=True,states=after,training_updates=0,test_access=0,DDIM_calls=0,
            raw_reused_windows=688,raw_generated_windows=0,wall_seconds=time.time()-start))
    protect();assert sources()==config['source_sha256']


def archive():
    from .report import build_report
    build_report()
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()==BRANCH
    subprocess.run(['git','add','protected_single_swap',str(DOC.relative_to(SOURCE))],check=True)
    subprocess.run(['git','diff','--cached','--check'],check=True)
    if subprocess.call(['git','diff','--cached','--quiet']):subprocess.run(['git','commit','-m','Archive HOTEL protected whole-scene single-swap development result'],check=True)
    subprocess.run(['git','push','-u','origin',BRANCH],check=True)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],text=True).split()[0];assert sha==remote
    write(RUN/'DELIVERY.json',dict(sha=sha,remote_verified_sha=remote,branch=BRANCH))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    os.chdir(SOURCE);RUN.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    if args.worker:export();return
    with (RUN/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);prepare()
        if args.prepare_only:status('PREPARED');return
        if not (RUN/'START.json').exists():immutable(RUN/'START.json',dict(unix=time.time()))
        if not (RUN/'PREDICTIONS_FROZEN.json').exists():
            with (RUN.parent/'auxiliary_gpu.lock').open('a') as aux:
                fcntl.flock(aux,fcntl.LOCK_EX)
                while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])<6144:
                    status('WAITING_AUXILIARY_GPU_RESOURCES');time.sleep(45)
                cmd=[sys.executable,'-m','protected_single_swap.run','--worker'];t=time.time()
                with (RUN/'export.log').open('a',buffering=1) as log:
                    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);write(RUN/'ACTIVE_CHILD.json',dict(pid=proc.pid));code=proc.wait()
                commands=read(RUN/'COMMANDS.json') if (RUN/'COMMANDS.json').exists() else []
                commands.append(dict(command=cmd,exit_code=code,start_unix=t,wall_seconds=time.time()-t));write(RUN/'COMMANDS.json',commands)
                if code:raise RuntimeError('search interrupted: resume same configuration and verified shards')
        status('CPU_SCORING_SEALED_ENDPOINTS')
        from .scoring import score
        score();protect()
        write(RUN/'WALL_TIME.json',dict(start_unix=read(RUN/'START.json')['unix'],end_unix=time.time(),wall_seconds=time.time()-read(RUN/'START.json')['unix']))
        decision=read(RUN/'DECISION.json')
        if decision['endpoint_gate_passed']:
            status('HOTEL_ENDPOINT_GATE_PASSED_REQUIRES_FIXED_TRAJECTORY_CONTINUATION');return
        status('ARCHIVING');archive();status('COMPLETE',training_updates=0,test_access=0,DDIM_calls=0)


if __name__=='__main__':
    try:main()
    except BaseException:status('INTERRUPTED_RESUME_SAME_EXPERIMENT',traceback=traceback.format_exc());raise
