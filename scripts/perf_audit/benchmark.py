"""Manual-only synthetic GPU benchmark. Refuses a live formal task or any GPU job.

No waiting, scheduler, queue, process signals, formal data, or formal-run output.
Profile mode and throughput mode are separate; profiler overhead is never speed.
"""
import argparse
import contextlib
import csv
import functools
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

SOURCE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SOURCE))
PERF_ROOT=SOURCE.parent/'RSJG_perf_outputs/bcep_v3_speed_audit_20261120'
LIVE=SOURCE.parent/'RSJG_EBJD_assets/experiments/zara2_bcep_v3_stage2_20261120'
SHAPES={'1x1':(1,), '4x8':(8,8,8,8), '1x57':(57,),
        'mixed16':(1,2,9,57,3,4,8,1,2,9,4,3,1,2,3,4)}
ARMS=['baseline','p0','p01','p012']
OPTIONAL=['map_no_checkpoint','bridge_no_checkpoint','both_no_checkpoint']


def admission(output, *, environ=None, status=None, compute_apps=None):
    """All checks precede importing torch or initializing any CUDA context."""
    environ=os.environ if environ is None else environ
    if environ.get('RUN_SYNTHETIC_PERF')!='true':
        raise PermissionError('explicit RUN_SYNTHETIC_PERF=true required')
    if any(environ.get(k)=='true' for k in ['RUN_ONE_ZARA2_V3','RUN_ONE_ZARA2']):
        raise PermissionError('formal-training authorization is not a benchmark authorization')
    status=json.loads((LIVE/'STATUS.json').read_text()) if status is None else status
    if status.get('status') not in ['COMPLETE','PAUSED_RESUMABLE']:
        raise RuntimeError('GPU_PROFILE_PENDING: formal job is not stopped; no wait or auto-schedule')
    if 'pid' in status and (Path('/proc')/str(status['pid'])).exists():
        raise RuntimeError('GPU_PROFILE_PENDING: formal controller still exists')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip() if compute_apps is None else compute_apps.strip()
    if apps:raise RuntimeError('GPU_PROFILE_PENDING: GPU is occupied; no benchmark started')
    output=Path(output).resolve()
    if not output.is_relative_to(PERF_ROOT.resolve()) or output==PERF_ROOT.resolve():
        raise ValueError('output must be a new child of the isolated performance directory')
    if output.exists():raise FileExistsError('benchmark output must be new; never overwrite a run')
    return output


def hash_state(value):
    import torch
    import numpy as np
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            a=x.detach().cpu().contiguous();h.update(str((a.dtype,tuple(a.shape))).encode());h.update(a.numpy().tobytes())
        elif isinstance(x,np.ndarray):
            h.update(str((x.dtype,x.shape)).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            for y in x:visit(y)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()


@contextlib.contextmanager
def profiler_labels(model,data,bank,observer):
    """Scopes are enabled only in profile mode; recomputation is included/labeled."""
    import torch
    import bcep_v2.stages as native_stages
    import bcep_v3.perf.stages as perf_stages
    import bcep_v2.joint as joint
    import bcep_v3.coupling as coupling
    def wrap(fn,name):
        @functools.wraps(fn)
        def call(*args,**kwargs):
            with torch.profiler.record_function(name):return fn(*args,**kwargs)
        return call
    with contextlib.ExitStack() as stack:
        targets=[(data,'batch','perf.cpu_data'),(bank,'batch','perf.cpu_candidates'),
                 (native_stages,'augment','perf.cpu_augment'),(perf_stages,'augment','perf.cpu_augment'),
                 (coupling,'reference_paths','perf.hcra_reference_paths'),(coupling,'pair_features','perf.hcra_pair_geometry'),
                 (model.bridge.encoder,'forward','perf.encoder'),(model.bridge.encoder.map_unet,'forward','perf.map_unet_forward_or_recompute'),
                 (model.coupling.csee,'forward','perf.csee'),(model.coupling.hcra,'forward','perf.hcra'),
                 (model.bridge.denoiser,'forward','perf.bridge_forward'),(model.bridge.denoiser.grg_ctf,'forward','perf.grg_ctf'),
                 (torch.Tensor,'backward','perf.backward')]
        for i,block in enumerate(model.bridge.denoiser.blocks):targets.append((block,'forward',f'perf.fjab_{i}_forward_or_recompute'))
        if observer:
            targets.extend([(observer,'capture_backward','perf.telemetry_gradient'),(observer,'step_done','perf.telemetry_delta_jsonl')])
        for obj,name,label in targets:stack.enter_context(patch.object(obj,name,wrap(getattr(obj,name),label)))
        original=joint.hard_assignment_indices;calls=0
        def hungarian(*a,**kw):
            nonlocal calls
            label='perf.hungarian_initial' if calls%2==0 else 'perf.hungarian_final';calls+=1
            with torch.profiler.record_function(label):return original(*a,**kw)
        stack.enter_context(patch.object(joint,'hard_assignment_indices',hungarian))
        yield


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['throughput','profile'],default='throughput')
    p.add_argument('--warmup',type=int,default=2);p.add_argument('--repetitions',type=int,default=5)
    p.add_argument('--include-checkpoint-arms',action='store_true');p.add_argument('--telemetry-off-control',action='store_true')
    p.add_argument('--shape',choices=list(SHAPES),action='append');p.add_argument('--arm',choices=ARMS+OPTIONAL+['telemetry_off'],action='append')
    a=p.parse_args()
    if a.warmup<1 or a.repetitions<5:raise ValueError('at least one warmup and five measured repetitions')
    arms=a.arm or ARMS+(OPTIONAL if a.include_checkpoint_arms else [])+(['telemetry_off'] if a.telemetry_off_control else [])
    shapes=a.shape or list(SHAPES)
    upper=len(arms)*len(shapes)*(a.warmup+(a.repetitions if a.mode=='throughput' else 1))
    if upper>300:raise ValueError('synthetic optimizer update cap 300')
    out=admission(a.output);out.mkdir(parents=True)
    import fcntl
    own_lock=(PERF_ROOT/'synthetic_gpu.lock').open('a')
    fcntl.flock(own_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)  # Only our performance directory; never a formal-run lock.
    # This import point is unreachable while the live training or any GPU job runs.
    import torch
    from bcep_v3.fixtures import SyntheticData,SyntheticBank,joint_config
    from bcep_v3.model import build
    from bcep_v3.perf.stages import joint_callbacks
    from bcep_v3.perf.telemetry import BatchedModuleTelemetry
    from bcep_v3.formal_telemetry import ModuleTelemetry
    from bcep_v3.perf.options import checkpoint_arm
    from bcep_v2.runtime import configure_cuda
    from bcep_v2.state import optimizer
    from bcep_joint.common import seed_all
    from bcep_joint.model import set_lr
    from bcep_formal.optimizations import finite_tensors
    configure_cuda();torch.set_num_threads(1)
    ledger=[];bench=[];states={};step_states={};completed_steps=0
    def write(name,x):(out/name).write_text(json.dumps(x,indent=2)+'\n')
    write('PROTOCOL.json',dict(synthetic_only=True,real_updates=0,max_synthetic_updates=upper,mode=a.mode,
        shapes=shapes,arms=arms,pixels=256,K=20,bridge_blocks=6,warmup=a.warmup,repetitions=a.repetitions,
        precision='FP32',SDPA='math',training_seed=3201,candidate_seed=2035,scheduler_epoch=1,scheduler_horizon=100,
        torch_version=torch.__version__,cuda_version=torch.version.cuda,gpu_name=torch.cuda.get_device_name(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
        auto_followup=False,statistical_claim='single invocation; intervals are exploratory, not cross-run confidence'))
    for shape in shapes:
        for arm in arms:
            status=json.loads((LIVE/'STATUS.json').read_text())
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()
            if status.get('status') not in ['COMPLETE','PAUSED_RESUMABLE'] or any(x.strip() and x.strip()!=str(os.getpid()) for x in apps):
                raise RuntimeError('GPU occupied by another task; abort without waiting or scheduling')
            seed_all(3201);config=joint_config();data=SyntheticData(SHAPES[shape],pixels=256);bank=SyntheticBank();model,_=build(config);model.cuda().train()
            observer=None if arm=='telemetry_off' else (ModuleTelemetry if arm=='baseline' else BatchedModuleTelemetry)(model,out/f'{shape}_{arm}_updates.jsonl',domain='synthetic')
            backward,_=joint_callbacks(model,data,'cuda',config,bank,bank,
                defer_diagnostics=arm not in ['baseline','p0','telemetry_off'],
                cpu_mask_check=arm not in ['baseline','p0','p01','telemetry_off'])
            order=torch.Generator().manual_seed(3201);view=torch.Generator().manual_seed(3202)
            opt=optimizer(model,1e-4);scheduler=set_lr(opt,1,100);cuda_samples=[];wall=[]
            def count(*args):
                nonlocal completed_steps
                completed_steps+=1
                ledger.append(dict(shape=shape,arm=arm,synthetic_update=completed_steps))
            opt.register_step_post_hook(count)
            def step():
                opt.zero_grad(set_to_none=True);metrics=backward(list(range(len(data.items))),order,view)
                if observer:observer.capture_backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                with torch.profiler.record_function('perf.adamw_step_and_post_hook'):opt.step()
                finite_tensors(model.parameters(),'post-update synthetic parameters')
                return metrics
            def state_record():
                return dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler,
                            cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                            order_rng=order.get_state(),view_rng=view.get_state())
            def verify_after_step():
                # Outside all timed/profile regions; the next pre-step state is this state.
                step_states.setdefault((shape,arm),[]).append(hash_state(state_record()))
            switches=dict(map_checkpoint=arm not in ['map_no_checkpoint','both_no_checkpoint'],bridge_checkpoint=arm not in ['bridge_no_checkpoint','both_no_checkpoint'])
            try:
                with checkpoint_arm(model,**switches), (profiler_labels(model,data,bank,observer) if a.mode=='profile' else contextlib.nullcontext()):
                    if observer:opt.register_step_post_hook(observer.step_done)
                    for _ in range(a.warmup):step();verify_after_step()
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    if a.mode=='profile':
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:step();torch.cuda.synchronize()
                        verify_after_step()
                        prof.export_chrome_trace(str(out/f'{shape}_{arm}_trace.json'))
                        (out/f'{shape}_{arm}_profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=120))
                        scopes=[dict(scope=e.key,calls=e.count,cpu_total_us=e.cpu_time_total,self_cpu_us=e.self_cpu_time_total,
                                     device_total_us=e.device_time_total,self_device_us=e.self_device_time_total)
                                for e in prof.key_averages() if e.key.startswith('perf.')]
                        with (out/f'{shape}_{arm}_PROFILE_SCOPES.csv').open('w',newline='') as f:
                            w=csv.DictWriter(f,fieldnames=list(scopes[0]));w.writeheader();w.writerows(scopes)
                    else:
                        for _ in range(a.repetitions):
                            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            torch.cuda.synchronize();t=time.perf_counter();start.record();step();end.record();torch.cuda.synchronize()
                            wall.append(time.perf_counter()-t);cuda_samples.append(start.elapsed_time(end))
                            verify_after_step()
                    state=state_record()
                    digest=hash_state(state);states[shape,arm]=digest
                    row=dict(shape=shape,arm=arm,status='MEASURED',state_sha256=digest,
                             bitwise_to_baseline=digest==states.get((shape,'baseline')) if (shape,'baseline') in states else None,
                             peak_allocated_bytes=torch.cuda.max_memory_allocated(),wall_seconds=wall,cuda_span_ms=cuda_samples)
                    if wall:
                        sorted_wall=sorted(wall);index=(len(wall)-1)*.9;lo=int(index);hi=min(lo+1,len(wall)-1)
                        row.update(p50_seconds=statistics.median(wall),p90_seconds=sorted_wall[lo]+(sorted_wall[hi]-sorted_wall[lo])*(index-lo),
                                   variance_seconds2=statistics.pvariance(wall),min_seconds=min(wall),max_seconds=max(wall))
                    if row['bitwise_to_baseline'] is False:row['exactness']='NON_BITWISE_OPTIONAL'
                    row['per_update_state_sha256']=step_states[shape,arm]
                    row['every_update_bitwise_to_baseline']=step_states[shape,arm]==step_states.get((shape,'baseline')) if (shape,'baseline') in step_states else None
                    if row['every_update_bitwise_to_baseline'] is False:row['exactness']='NON_BITWISE_OPTIONAL'
                    bench.append(row);del state
            except torch.cuda.OutOfMemoryError:
                bench.append(dict(shape=shape,arm=arm,status='OOM_OPTIONAL_DISABLED'))
            finally:
                write('RESULTS.json',bench);write('UPDATE_LEDGER.json',ledger)
                write('TOTAL_UPDATES.json',dict(synthetic_optimizer_updates=completed_steps,real_optimizer_updates=0))
                del model,opt,observer,backward;__import__('gc').collect();torch.cuda.empty_cache()
    if a.mode=='throughput':
        rows=[]
        for r in bench:
            if r['status']!='MEASURED':continue
            base=next((x for x in bench if x['shape']==r['shape'] and x['arm']=='baseline' and x['status']=='MEASURED'),None)
            rows.append({k:r[k] for k in ['shape','arm','p50_seconds','p90_seconds','variance_seconds2','min_seconds','max_seconds','peak_allocated_bytes','bitwise_to_baseline']})
            rows[-1]['p50_time_reduction_percent']=100*(base['p50_seconds']-r['p50_seconds'])/base['p50_seconds'] if base else None
        if rows:
            with (out/'BENCHMARK.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Synthetic benchmark completed; no formal queue or follow-up launched.')


if __name__=='__main__':main()
