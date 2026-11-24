"""HOTEL dev raw endpoints once, 16 frozen uniform sets, global seal, CPU scoring."""
import argparse,fcntl,os,subprocess,sys,time,traceback
from contextlib import contextmanager
import numpy as np
import torch
from bcep_joint.common import local_seed
from bcep_formal.data import FormalData
from joint_flow_fourfold.common import immutable
from joint_flow_fourfold.inputs import state_hash
from joint_marginal_residual.assets import load_inputs,load_teacher
from joint_marginal_residual.run import protect
from joint_flow_dependency.inference import encode_observed,sample_endpoints
from bcep_v2.provider import GEOMETRY
from .common import *


def status(value,**kwargs):
    d=dict(status=value,unix=time.time(),**kwargs);write(RUN/'STATUS.json',d);print(d,flush=True)


@contextmanager
def prediction_guard():
    old=FormalData.item;get=np.lib.npyio.NpzFile.__getitem__
    def guarded(self,index,purpose,truth=True):
        if purpose!='selection_dev' or truth:raise RuntimeError('only observed HOTEL dev is permitted')
        return old(self,index,purpose,truth)
    def guarded_get(self,key):
        if key in ('future','future.npy'):raise RuntimeError('future forbidden before endpoint seal')
        return get(self,key)
    FormalData.item=guarded;np.lib.npyio.NpzFile.__getitem__=guarded_get
    try:yield
    finally:FormalData.item=old;np.lib.npyio.NpzFile.__getitem__=get


def random_ids(member,rep):
    seed=local_seed('hotel-endpoint-uniform-whole-scenes-v1',canonical_hash(member),2035,int(rep))
    return torch.randperm(256,generator=torch.Generator().manual_seed(seed))[:20].sort().values.numpy()


def prepare():
    RUN.mkdir(parents=True,exist_ok=True);protect()
    with prediction_guard():
        inputs,model,provider,dep,identity=load_teacher('hotel','cpu')
        assert identity['teacher_epoch']==19
        ids=inputs.split['selection_dev_indices'];ns=[len(inputs.split['members'][i]['agent_ids']) for i in ids]
        assert (len(ids),sum(ns))==(688,5203)
        rows=[]
        for pos,i in enumerate(ids):
            m=inputs.split['members'][i]
            rows.append(dict(window=f'dev_{pos+1:04d}',index=i,n=ns[pos],member_sha256=canonical_hash(m),
                random_ids=np.stack([random_ids(m,r) for r in range(16)]).tolist()))
        immutable(RUN/'RANDOM_IDS.json',dict(seed=2035,repeats=16,rows=rows))
    config=dict(base_commit=BASE,fold='hotel',purpose='selection_dev',seed=2035,M=256,K=20,repeats=16,
        windows=688,agents=5203,identity=identity,reference=REFERENCE,
        reference_file_sha256=digest(ROOT/'RSJG_EBJD_assets/experiments/joint_flow_fourfold_confirmation_20261122/hotel/dependency/dev/epoch_019_groups.json'),
        random_ids_sha256=digest(RUN/'RANDOM_IDS.json'),source_sha256=sources(),
        native_disabled_optimizations='geometry,hungarian,logging,modulation,proposal,schedule',
        precision='CUDA deterministic FP32 endpoint generation and CandidateHead; CPU FP32 endpoint errors; float64 expectation and accumulation',
        replay_tolerance=dict(atol=1e-6,rtol=1e-6),training_updates=0,test_access=0,DDIM_calls=0,
        Bstar_decision='retain B as cross-fold main model; close post-bridge Bstar residual head; retain original pre-bridge CandidateHead',
        raw_cache='No identity-matching real raw256 HOTEL dev coordinates exist; regenerate once/window; never reconstruct from K20 IDs or summaries',
        uniform_sampling='16 sorted uniform-without-replacement whole-scene sets from torch.randperm; stable hash of full member,seed2035,repeat; frozen before labels',
        expectation='C(M-r,K-1)/C(M,K), r=1..M-K+1; exact K20 marginal and joint expectation before CandidateHead',
        head='frozen original CandidateHead called separately once per native/random K20 set; do not reuse corrected endpoints across sets',
        motion=dict(speed='sum of seven observed consecutive distances /2.8 seconds',threshold_m_per_s=.2,low_inclusive=True),
        differences='native20 minus uniform20; Bstar historical difference is Bstar minus B and is not an experimental branch here',
        metrics='endpoint only: L,G,J window equal weight, formal minFDE valid-agent weighted; deltaJ=deltaL+deltaG',
        resources='existing auxiliary_gpu.lock queue, one worker, 40% allocator cap, >=6GiB free before start; CPU scoring two threads',
        known_test_background='HOTEL already-seen test results motivated this hypothesis; no test reads, training, speed threshold search or per-fold B/Bstar selection')
    immutable(RUN/'RUN_CONFIG.json',config);return config


def validate_shard(path,n):
    with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    shapes=dict(raw=(256,n,2),native_pre=(20,n,2),native_post=(20,n,2),random_post=(16,20,n,2),observed=(n,8,2))
    for k,s in shapes.items():assert a[k].shape==s and a[k].dtype==np.float32 and np.isfinite(a[k]).all()
    assert a['valid'].shape==(n,) and a['valid'].all()
    assert a['native_ids'].shape==(20,) and len(np.unique(a['native_ids']))==20
    assert a['random_ids'].shape==(16,20)
    for ids in a['random_ids']:assert len(np.unique(ids))==20 and ids.min()>=0 and ids.max()<256
    assert np.array_equal(a['native_pre'],a['raw'][a['native_ids']])
    return a


@torch.no_grad()
def export():
    from bcep_v2.runtime import configure_cuda
    c=read(RUN/'RUN_CONFIG.json');assert sources()==c['source_sha256'];protect()
    torch.set_num_threads(2);configure_cuda();torch.cuda.set_per_process_memory_fraction(.4)
    os.environ['RSJG_DISABLE_OPTIMIZATIONS']=c['native_disabled_optimizations']
    rows=[];start=time.time();(RUN/'predictions').mkdir(exist_ok=True)
    with prediction_guard():
        inputs,model,provider,dep,identity=load_teacher('hotel','cuda');assert identity==c['identity']
        modules=dict(backbone=model,provider=provider,dependence=dep);before={k:state_hash(m) for k,m in modules.items()}
        # Decoder must never be entered in this endpoint-only diagnostic.
        def deny_decode(*args,**kwargs):raise RuntimeError('DDIM/bridge decoder prohibited')
        model.bridge.forward=deny_decode;model.bridge.denoiser.forward=deny_decode
        for pos,item in enumerate(read(RUN/'RANDOM_IDS.json')['rows']):
            index=item['index'];n=item['n'];p=RUN/'predictions'/f"{item['window']}.npz";side=p.with_suffix('.json')
            expected=dict(window=item['window'],index=index,n=n,file=str(p.relative_to(RUN)),member_sha256=item['member_sha256'],config_sha256=digest(RUN/'RUN_CONFIG.json'))
            if side.exists():
                row=read(side);assert {k:row[k] for k in expected}==expected and row['sha256']==digest(p)
                a=validate_shard(p,n);assert np.array_equal(a['random_ids'],item['random_ids'])
            else:
                tick=time.monotonic();host=inputs.data.batch([index],'selection_dev',truth=False)
                assert host.valid.all() and not host.future.count_nonzero()
                batch=host.to('cuda');context,features=encode_observed(provider,batch)
                endpoints=sample_endpoints(provider,dep,batch,context,features,seed=2035)
                assert endpoints['correction_calls']==1
                raw=endpoints['raw'];random=[]
                for ids in item['random_ids']:
                    goals=raw[torch.tensor(ids,device='cuda')]
                    corrected,_=provider.head(provider.base,batch.observed[0],batch.semantic_maps[0],batch.valid[0],
                        goals,context,features,dict(layout=GEOMETRY));random.append(corrected.cpu().numpy())
                saved=torch.load(PRIOR_REPLAY/f'{pos:05d}.pt',map_location='cpu',weights_only=False)
                corrected=endpoints['corrected'][0].cpu();prior_end=saved['B'][:,:,-1]
                torch.testing.assert_close(corrected,prior_end,atol=1e-6,rtol=1e-6)
                a=dict(raw=raw.cpu().numpy(),native_ids=endpoints['raw_indices'].cpu().numpy(),
                    native_pre=endpoints['selected'].cpu().numpy(),native_post=corrected.numpy(),
                    random_ids=np.asarray(item['random_ids'],dtype=np.int64),random_post=np.stack(random),
                    observed=host.observed[0].numpy(),valid=host.valid[0].numpy())
                if p.exists():
                    previous=validate_shard(p,n);assert all(np.array_equal(previous[k],v) for k,v in a.items())
                else:
                    tmp=p.with_suffix('.tmp.npz');np.savez_compressed(tmp,**a);tmp.replace(p)
                validate_shard(p,n)
                row=dict(expected,sha256=digest(p),seconds=time.monotonic()-tick,raw_generation_calls=1,
                    CandidateHead_calls=17,DDIM_calls=0,native_endpoint_replay_max_abs_m=float((corrected-prior_end).abs().max()))
                write(side,row)
            rows.append(row)
            if (pos+1)%25==0 or pos==687:status('OBSERVED_ENDPOINTS',windows=pos+1,total=688,elapsed_seconds=time.time()-start)
        after={k:state_hash(m) for k,m in modules.items()};assert after==before
        for row in rows:assert digest(RUN/row['file'])==row['sha256'];validate_shard(RUN/row['file'],row['n'])
        seal=dict(rows=rows,seal_unix=time.time(),config_sha256=digest(RUN/'RUN_CONFIG.json'),future_reads=0,
            random_ids_sha256=digest(RUN/'RANDOM_IDS.json'),weights_unchanged=True,states=after,raw_reused_windows=0,
            raw_generated_windows=688,CandidateHead_calls=688*17,DDIM_calls=0,wall_seconds=time.time()-start)
        immutable(RUN/'PREDICTIONS_FROZEN.json',seal)
    protect();assert sources()==c['source_sha256']


def archive():
    from .report import build_report
    build_report()
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()==BRANCH
    subprocess.run(['git','add','hotel_endpoint_diagnostic',str(DOC.relative_to(SOURCE))],check=True)
    subprocess.run(['git','diff','--cached','--check'],check=True)
    if subprocess.call(['git','diff','--cached','--quiet']):subprocess.run(['git','commit','-m','Archive HOTEL frozen endpoint selection diagnostic and close residual-head route'],check=True)
    subprocess.run(['git','push','-u','origin',BRANCH],check=True)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],text=True).split()[0];assert sha==remote
    write(RUN/'DELIVERY.json',dict(sha=sha,remote_verified_sha=remote,branch=BRANCH))


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    os.chdir(SOURCE);RUN.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    if args.worker:export();return
    with (RUN/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);config=prepare()
        if args.prepare_only:status('PREPARED');return
        if not (RUN/'START.json').exists():immutable(RUN/'START.json',dict(unix=time.time()))
        if not (RUN/'PREDICTIONS_FROZEN.json').exists():
            with (RUN.parent/'auxiliary_gpu.lock').open('a') as aux:
                fcntl.flock(aux,fcntl.LOCK_EX)
                for attempt in range(1,4):
                    while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])<6144:
                        status('WAITING_EXISTING_AUX_GPU_QUEUE');time.sleep(45)
                    cmd=[sys.executable,'-m','hotel_endpoint_diagnostic.run','--worker'];t=time.time()
                    with (RUN/'export.log').open('a',buffering=1) as log:
                        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);write(RUN/'ACTIVE_CHILD.json',dict(pid=proc.pid));code=proc.wait()
                    commands=read(RUN/'COMMANDS.json') if (RUN/'COMMANDS.json').exists() else []
                    commands.append(dict(command=cmd,exit_code=code,attempt=attempt,start_unix=t,wall_seconds=time.time()-t));write(RUN/'COMMANDS.json',commands)
                    if code==0:break
                    if attempt==3:raise RuntimeError('export failed; resume same diagnostic')
                    time.sleep(15)
        status('CPU_SCORING_SEALED_ENDPOINTS')
        from .scoring import score
        score();protect()
        write(RUN/'WALL_TIME.json',dict(start_unix=read(RUN/'START.json')['unix'],end_unix=time.time(),wall_seconds=time.time()-read(RUN/'START.json')['unix']))
        status('ARCHIVING');archive();status('COMPLETE',training_updates=0,test_access=0,DDIM_calls=0)

if __name__=='__main__':
    try:main()
    except BaseException:status('INTERRUPTED_RESUME_SAME_DIAGNOSTIC',traceback=traceback.format_exc());raise
