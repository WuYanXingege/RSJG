"""Bounded engineering gate. Only synthetic optimizer updates are permitted."""
import argparse,copy,json,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bcep_pilot_runtime import configure,environment,assert_gpu_idle,script_module,ROOT
import numpy as np
import torch,yaml
from bcep.contracts import load_locked_split,code_identity,digest,check_item
from bcep.provider import BCEP
from bcep.pilot_runner import run_pilot,atomic_json
from ebjd.data import SceneManifestDataset


def compare(a,b,rtol=1e-4,atol=1e-5):
    a=a.detach().cpu();b=b.detach().cpu()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise AssertionError('nonfinite comparison')
    difference=(a-b).abs();scale=torch.maximum(a.abs(),b.abs()).clamp_min(1e-12)
    return dict(max_absolute=float(difference.max()),max_relative=float((difference/scale).max()),
        exactly_equal=torch.equal(a,b),within_tolerance=torch.allclose(a,b,rtol=rtol,atol=atol),rtol=rtol,atol=atol)


def tree_difference(a,b):
    records=[]
    def walk(x,y,path):
        if isinstance(x,torch.Tensor):
            if x.is_floating_point():records.append(dict(path=path,**compare(x,y,1e-6,1e-7)))
            else:assert torch.equal(x,y),path
        elif isinstance(x,np.ndarray):assert np.array_equal(x,y),path
        elif isinstance(x,dict):
            assert x.keys()==y.keys(),path
            for k in x:walk(x[k],y[k],path+'.'+str(k))
        elif isinstance(x,(list,tuple)):
            assert len(x)==len(y),path
            for i,(xx,yy) in enumerate(zip(x,y)):walk(xx,yy,path+'.'+str(i))
        else:assert x==y,path
    walk(a,b,'state')
    return dict(exactly_equal=all(x['exactly_equal'] for x in records),within_tolerance=all(x['within_tolerance'] for x in records),
        floating_tensors=len(records),max_absolute=max((x['max_absolute'] for x in records),default=0),
        max_relative=max((x['max_relative'] for x in records),default=0),
        largest_differences=sorted(records,key=lambda x:x['max_absolute'],reverse=True)[:5],nonfloating_equal=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());root=Path(plan['asset_root']);output=root/'gate'
    report_path=root/'reports/GPU_GATE_REPORT.json'
    if report_path.exists():raise RuntimeError('one gate only; existing report must not be overwritten')
    cpu_suite=ET.parse(output/'CPU_REGRESSION.xml').getroot().find('testsuite')
    assert int(cpu_suite.attrib['tests'])==25 and cpu_suite.attrib['failures']==cpu_suite.attrib['errors']=='0'
    cpu_seconds=float(cpu_suite.attrib['time'])
    started=time.monotonic();report=dict(cpu_regression_seconds=cpu_seconds,cpu_tests_passed=25,status='RUNNING',real_optimizer_updates=0,synthetic_optimizer_updates=0,checks={},limits=dict(cpu_gpu=dict(rtol=1e-4,atol=1e-5),gpu_resume=dict(rtol=1e-6,atol=1e-7)))
    atomic_json(report,report_path)
    try:
        assert_gpu_idle();report['precision']=configure()
        assert torch.cuda.is_available(),'BLOCKED_RESOURCE: CUDA unavailable'
        report['environment']=environment();torch.cuda.reset_peak_memory_stats()
        cfg=yaml.safe_load(a.config.read_text());split,contract,members=load_locked_split(ROOT/cfg['split_lock'],ROOT/cfg['contract'])
        assert len(split['train_indices'])==3790 and len(split['dev_indices'])==320
        # Choose before inference, purely by actual cardinality and locked index.
        allowed=sorted(split['train_indices']+split['dev_indices'])
        selected=[next(i for i in allowed if len(members[i]['agent_ids'])==1),
                  next(i for i in allowed if len(members[i]['agent_ids'])>=2),
                  min(allowed,key=lambda i:(-len(members[i]['agent_ids']),i))]
        report['fixed_real_windows']=[members[i] for i in selected]
        report['source_identity']=code_identity();report['config_sha256']=digest(a.config)
        atomic_json(report,report_path)
        def checkpoint():
            torch.cuda.synchronize()
            if time.monotonic()-started+cpu_seconds>600:raise TimeoutError('GPU gate wall budget exhausted')
            atomic_json(report,report_path)
        torch.manual_seed(3101);cpu=BCEP().eval()
        cpu.scales.copy_(torch.tensor(split['scales']));cpu.scales_fitted.fill_(True)
        with torch.no_grad():
            for net in cpu.flow.nets:net[-1].weight.normal_(0,.02)
        gpu=copy.deepcopy(cpu).cuda().eval()
        obs=torch.randn(3,8,2)*.2;maps=torch.rand(3,6,32,48);valid=torch.ones(3,dtype=torch.bool);future=torch.randn(3,12,2)*.3
        with torch.no_grad():
            context_c=cpu.encoder(obs,maps,valid,.4);context_g=gpu.encoder(obs.cuda(),maps.cuda(),valid.cuda(),.4)
            report['checks']['context_cpu_gpu']=compare(context_c,context_g)
            cc=cpu.loss_components(obs,maps,valid,future);cg=gpu.loss_components(obs.cuda(),maps.cuda(),valid.cuda(),future.cuda())
            report['checks']['composite_cpu_gpu']=compare(cc['loss_total'],cg['loss_total'])
            report['checks']['world_NLL_cpu_gpu']=compare(cc['nll_final_sum'],cg['nll_final_sum'])
            atoms_c,_=cpu.sample_support(obs,maps,valid,'synthetic',['a','b','c'],2035)
            atoms_g,_=gpu.sample_support(obs.cuda(),maps.cuda(),valid.cuda(),'synthetic',['a','b','c'],2035)
            report['checks']['world_samples_cpu_gpu']=compare(atoms_c,atoms_g)
            latent=torch.tensor([[.3,-.8],[1.1,.2],[-.2,.1]],device='cuda')
            y,ld=gpu.flow.transform(latent,context_g,4.8);z,il=gpu.flow.transform(y,context_g,4.8,inverse=True)
            report['checks']['inverse']=compare(latent,z)
            report['checks']['inverse_logdet']=compare(ld,-il)
        jac=torch.autograd.functional.jacobian(lambda v:gpu.flow.transform(v[None],context_g[:1],4.8)[0][0],latent[0])
        report['checks']['jacobian']=compare(torch.linalg.slogdet(jac)[1],ld[0])
        loss=gpu.loss(obs.cuda(),maps.cuda(),valid.cuda(),future.cuda());loss.backward()
        grads={name:dict(finite=bool(torch.isfinite(par.grad).all()),norm=float(par.grad.norm())) for name,par in gpu.named_parameters() if par.grad is not None}
        assert all(x['finite'] for x in grads.values())
        for key in ['encoder.edge_bias.weight','encoder.edge_value.weight']:assert grads[key]['norm']>0,key
        report['gradient_checks']=grads
        checkpoint()
        assert all(x['within_tolerance'] for x in report['checks'].values()),'CPU/GPU or flow gate failed'
        # Exact same production runner, synthetic input only. Exercise CUDA RNG as well.
        fixture=script_module('tests/test_bcep_prepilot.py');ds,syn_split,syn_cfg,identity=fixture.runner_fixture();syn_cfg.update(device='cuda:0',budget_seconds=600)
        for item in ds.items:item['semantic_maps']=torch.rand(len(item['observed']),6,32,48)
        class DeviceRNGDataset:
            def __getitem__(self,index):
                value=ds[index];value['observed']=value['observed']+torch.rand((),device='cuda').cpu()*.001;return value
        ds_gpu=DeviceRNGDataset();continuous=[];resumed=[]
        run_pilot(ds_gpu,syn_split,syn_cfg,identity,output/'continuous',stop_after_updates=8,on_update=lambda i,u:continuous.append((i,u)))
        report['synthetic_optimizer_updates']+=8;checkpoint()
        run_pilot(ds_gpu,syn_split,syn_cfg,identity,output/'resumed',stop_after_updates=3,on_update=lambda i,u:resumed.append((i,u)))
        report['synthetic_optimizer_updates']+=3;checkpoint()
        run_pilot(ds_gpu,syn_split,syn_cfg,identity,output/'resumed',resume=True,stop_after_updates=5,on_update=lambda i,u:resumed.append((i,u)))
        report['synthetic_optimizer_updates']+=5
        left=torch.load(output/'continuous/last.pt',map_location='cpu',weights_only=False)
        right=torch.load(output/'resumed/last.pt',map_location='cpu',weights_only=False)
        assert continuous==resumed and len(continuous)==8
        fields=['model','optimizer','scheduler','rng','order','epoch','completed_epoch','update','train_cursor','dev_cursor','best','stale','selected_epoch','train_accumulator','dev_accumulator']
        report['gpu_resume']=tree_difference({k:left[k] for k in fields},{k:right[k] for k in fields})
        assert report['gpu_resume']['within_tolerance'],'GPU restore exceeds frozen tolerance'
        report['gpu_resume']['sample_trace']=continuous
        checkpoint()
        dataset=SceneManifestDataset(split['source_manifest']);report['real_interface']=[]
        with torch.no_grad():
            for index in selected:
                item=dataset[index];check_item(item,members[index]);o=item['observed'];v=torch.ones(len(o),dtype=torch.bool);m=item['semantic_maps']
                tc=cpu.encoder(o,m,v,.4);tg=gpu.encoder(o.cuda(),m.cuda(),v.cuda(),.4)
                ac,_=cpu.sample_support(o,m,v,item['scene_id'],item['agent_ids'],2035)
                ag,_=gpu.sample_support(o.cuda(),m.cuda(),v.cuda(),item['scene_id'],item['agent_ids'],2035)
                iid=ag[:20].clone();coverage,meta=gpu.select_support(ag,v.cuda(),item['agent_ids'])
                same_cpu,meta_cpu=cpu.select_support(ag.cpu(),v,item['agent_ids'])
                assert torch.equal(iid.cpu(),ag[:20].cpu()) and torch.equal(coverage.cpu(),same_cpu)
                assert meta==meta_cpu
                # GT enters likelihood only after both candidate sets have been fixed.
                lc=cpu.loss_components(o,m,v,item['future']);lg=gpu.loss_components(o.cuda(),m.cuda(),v.cuda(),item['future'].cuda())
                checks=dict(context=compare(tc,tg),world_samples=compare(ac,ag),world_NLL=compare(lc['nll_final_sum'],lg['nll_final_sum']))
                report['real_interface'].append(dict(index=index,agents=len(o),map_shape=list(m.shape),status='NONFORMAL',checks=checks,shared_support_exact=True))
                checkpoint()
                assert all(x['within_tolerance'] for x in checks.values()),'real CPU/GPU check failed'
        report['status']='PASS'
    except Exception as e:
        report['status']='BLOCKED_RESOURCE' if 'BLOCKED_RESOURCE' in str(e) else 'BLOCKED'
        report['failure']=str(e);report['traceback']=traceback.format_exc()
    finally:
        report['wall_seconds']=time.monotonic()-started
        report['total_preflight_seconds_including_cpu_suite']=report['wall_seconds']+cpu_seconds
        if torch.cuda.is_initialized():
            torch.cuda.synchronize();report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        atomic_json(report,report_path);print(json.dumps({k:report[k] for k in ['status','wall_seconds','real_optimizer_updates','synthetic_optimizer_updates']}),flush=True)
    if report['status']!='PASS':raise SystemExit(2)

if __name__=='__main__':main()
