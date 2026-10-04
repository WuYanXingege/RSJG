#!/usr/bin/env python3
"""Four slots, two original message calls per slot; gated extra calls only inside T1."""
import argparse
import contextlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tools.jdv2_stage_a_bank import REPO, CHECKPOINT, CHECKPOINT_SHA, CONFIG, CONFIG_SHA, require, sha256, write_json, code_provenance
from tools.jdv2_restart_observer import Recorder, compare_tensor, state_fingerprint
from tools.jdv2_restart_probe import save_torch, execution_settings
from tools.jdv2_social_local import fingerprint
from tools.jdv2_message_observer import MessageObserver, compare_traces

BASE = 'a13eae9e1bd142b95b1cb0ed26667fec0f56cae4'
PRIOR = REPO / 'docs/joint_dependency_v2/reviews/2026-10-04_0ffdf7c_social_encoder_local'
OLD = REPO / 'outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/social_encoder_local_634a467'
ROOT = REPO / 'outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/message_layer_local_a13eae9'
FIELDS = ('agent_feat','edge_index','edge_feat','edge_weight')
SOURCES = ('tools/jdv2_message_observer.py','tools/jdv2_message_diagnostic.py','tools/jdv2_message_cpu.py','tools/jdv2_message_analyze.py')


def read(path):
    return json.loads(path.read_text())


def authenticate():
    require(not subprocess.check_output(['git','diff',BASE,'--','src','configs']), 'Production/config changed', 'BLOCKED_INPUT')
    code = code_provenance()
    require(sha256(REPO/CHECKPOINT)==CHECKPOINT_SHA and sha256(REPO/CONFIG)==CONFIG_SHA,'Frozen hash mismatch','BLOCKED_INPUT')
    for name,digest in read(PRIOR/'ARTIFACT_HASHES.json')['files'].items():
        require(sha256(PRIOR/name)==digest,'Prior archive hash: '+name,'BLOCKED_INPUT')
    manifest=read(PRIOR/'ARTIFACT_MANIFEST.json')['files']
    files={}
    for p in ('T0','T1'):
        for name in ('TRACE.pt','TRACE_CONTRACTS.json','INPUT_MANIFEST.json','INITIALIZED.json','VALIDATION.json','RNG.pt','COMPLETE.json'):
            key=p+'/'+name
            require(sha256(OLD/key)==manifest[key]['sha256'],'Prior file hash: '+key,'BLOCKED_INPUT')
            files[key]=manifest[key]
    tapes={p:torch.load(OLD/p/'TRACE.pt',weights_only=True,map_location='cpu') for p in ('T0','T1')}
    metas={p:read(OLD/p/'TRACE_CONTRACTS.json')['tensors'] for p in tapes}
    contracts={}
    prefix='message_layer_0/input/'
    for name,shape in zip(FIELDS,((9,128),(2,36),(36,14),(36,))):
        key=prefix+name
        for p in tapes:
            t,m=tapes[p][key],metas[p][key]
            require(m['valid_snapshot'] and m['version_at_capture']==m['version_at_dump'],'Invalid prior snapshot','BLOCKED_INPUT')
            require(tuple(t.shape)==shape and t.dtype==(torch.int64 if name=='edge_index' else torch.float32),'Input shape/dtype','BLOCKED_INPUT')
            require(all(fingerprint(t)[f]==m[f] for f in fingerprint(t)),'Unsupported input layout reconstruction','BLOCKED_INPUT')
        require(compare_tensor(tapes['T0'][key],tapes['T1'][key])['bitwise_equal'],'T0/T1 input mismatch','BLOCKED_INPUT')
        contracts[name]={'T0':metas['T0'][key],'T1':metas['T1'][key],'saved_cpu':fingerprint(tapes['T0'][key])}
    require(contracts['agent_feat']['T0']['raw_sha256']=='058bf0d81ce4260309e76e22ca91c1e1eee10545de0dd573936ef0bb9394656d','Wrong historical GRU output','BLOCKED_INPUT')
    for p in tapes:
        require(len({metas[p][prefix+k]['alias_group'] for k in FIELDS})==4,'Shared fields require explicit reconstruction','BLOCKED_INPUT')
        require(len({tapes[p][prefix+k].untyped_storage()._cdata for k in FIELDS})==4,'Unexpected stored alias','BLOCKED_INPUT')
    from src.models.social_encoder import _SocialMessageLayer
    constructor={'hidden_dim':128,'edge_dim':14,'dropout':0.0}
    layer=_SocialMessageLayer(**constructor)
    checkpoint=torch.load(REPO/CHECKPOINT,weights_only=False,map_location='cpu')
    require(checkpoint['epoch']==13 and checkpoint['training_stage']=='joint_goal' and checkpoint['latent_objective']=='strict_no_z','Wrong checkpoint route','BLOCKED_INPUT')
    state_prefix='social_encoder.message_layers.0.'
    extracted={k[len(state_prefix):]:v for k,v in checkpoint['model_state_dict'].items() if k.startswith(state_prefix)}
    layer.load_state_dict(extracted,strict=True);layer.eval()
    loaded=state_fingerprint(layer)
    for p in tapes:
        ref=read(OLD/p/'INPUT_MANIFEST.json')
        require(ref['constructor']=={'hidden_dim':128,'output_dim':128,'edge_dim':14,'num_message_layers':1,'dropout':0.0,'dt':0.4},'Constructor reference mismatch','BLOCKED_INPUT')
        previous={k[len('message_layers.0.'):]:v for k,v in ref['encoder_state'].items() if k.startswith('message_layers.0.')}
        require(set(loaded)==set(previous),'Message state keys','BLOCKED_INPUT')
        for k,v in loaded.items():
            require(all(v[f]==previous[k][f] for f in ('dtype','shape','sha256')),'Message weight bytes changed','BLOCKED_INPUT')
    return {k:tapes['T0'][prefix+k] for k in FIELDS},layer,{
        'base_commit':BASE,'production_src_tree':subprocess.check_output(['git','rev-parse','HEAD:src'],text=True).strip(),
        'code':code,'prior_files_authenticated':files,'checkpoint_sha256':CHECKPOINT_SHA,'config_sha256':CONFIG_SHA,
        'inputs':contracts,'selected_source':'T0 only; T1 authenticates equality, never used as alternative input',
        'layout':'all four contiguous offset0, distinct groups; no reconstruction; source/target remain original edge_index views in original forward',
        'constructor':constructor,'strict_prefix':state_prefix,'strict_load':True,'layer_state':loaded,
        'settings_reference':read(OLD/'T0/INITIALIZED.json')['settings'],
        'unrestored':['CUDA addresses','allocator','historical stream/workspace','historical message-entry RNG']}


def controlled(layer,live,gate,out,counts,resource_gate):
    """Only chosen first region. Allocation/copy/CPU inspection costs are explicit."""
    region=gate['eligible_region'];record={'region':region,'calls':[],'preparation':{'zero_allocations':0,'prestate_clones':0},
        'observer':'OFF; independent calls with CPU pre/post hashes and sync; not canonical execution context',
        'status':'NOT_REPRODUCED_WITHIN_BUDGET'}
    outputs={}
    def call(stage,index,input_record,fn):
        resource_gate()
        require(counts['extra_attempted']<8,'Extra budget exhausted')
        counts['extra_attempted']+=1;counts['extra_unknown_inflight']=1
        key=f'{stage}/{index}'
        write_json(out/f'extra_budget/{counts["extra_attempted"]}_attempt.json',dict(counts))
        result=fn()
        counts['extra_completed']+=1;counts['extra_unknown_inflight']=0
        write_json(out/f'extra_budget/{counts["extra_attempted"]}_complete.json',dict(counts))
        cpu=result.detach().cpu().clone()
        row={'key':key,'input':input_record,'output':fingerprint(result),'return_is_mutated_accumulator':stage in ('A','B')}
        outputs[key]=cpu;record['calls'].append(row)
        return result,cpu
    def run_stage(stage,idx,data,seed):
        states=[];cpus=[]
        for index in range(1,5):
            if stage=='A':
                accumulator=torch.zeros_like(seed)
                record['preparation']['zero_allocations']+=1
            else:
                accumulator=seed.clone()
                record['preparation']['prestate_clones']+=1
            entrance={'accumulator':fingerprint(accumulator),'index':fingerprint(idx),'source':fingerprint(data),'dim':0,
                      'origin':'diagnostic zero per original zeros_like definition' if stage=='A' else 'clone of prespecified first A output, not an original-run snapshot'}
            result,cpu=call(stage,index,entrance,lambda:accumulator.index_add_(0,idx,data))
            require(result is accumulator,'index_add return identity changed')
            states.append(result);cpus.append(cpu)
            if index>1 and not compare_tensor(cpus[0],cpu)['bitwise_equal']:
                record['variation_stage']=stage
                record['first_vs_different']=compare_tensor(cpus[0],cpu)
                return True,states[0]
        return False,states[0]
    with torch.no_grad(),torch.autocast(device_type='cuda',enabled=False):
        if region in ('feature_accumulation','normalizer'):
            idx_source,idx_target=live['weighted/source_index'],live['weighted/target_index']
            if region=='feature_accumulation':
                source,target=live['weighted/source'],live['weighted/target']
                template=live['entry/agent_feat']
            else:
                source=target=live['entry/edge_weight']
                template=live['normalizer/after_both_adds']
            varied,first=run_stage('A',idx_source,source,template)
            if not varied:
                varied,_=run_stage('B',idx_target,target,first)
            record['status']='CONTROLLED_PRIMITIVE_VARIATION_REPRODUCED' if varied else 'NOT_REPRODUCED_WITHIN_BUDGET'
        else:
            module=layer.message_mlp if region.startswith('message_mlp/') else getattr(layer,region)
            entrance=live[region+'/input']
            first=None;varied=False
            for index in range(1,9):
                _,cpu=call('module',index,{'actual_input':fingerprint(entrance)},lambda:module(entrance))
                if first is None:first=cpu
                elif not compare_tensor(first,cpu)['bitwise_equal']:
                    record['first_vs_different']=compare_tensor(first,cpu);varied=True;break
            record['status']='CONTROLLED_SUBMODULE_VARIATION_REPRODUCED' if varied else 'NOT_REPRODUCED_WITHIN_BUDGET'
    record['preparation']['extra_context_cost']='Independent allocations/clones, retained independent returns, CPU input hashing and post-call copies synchronize execution; all outside original message batch.'
    save_torch(out/'CONTROLLED.pt',outputs)
    write_json(out/'CONTROLLED.json',record)
    return record


def run(process,uuid):
    from tools.jdv2_export_stage_a_bank import gpu_snapshot,require_idle,DESKTOP_EXECUTABLES
    from tools.jdv2_stage_b_v1_failure_mechanism_audit import rng_snapshot,rng_restore
    from tools.jdv2_stage_b_identity_contract_audit import _rng_equal
    out=ROOT/process;out.mkdir(parents=True,exist_ok=False)
    counts={'processes_started':1,'initialization_attempted':0,'initialization_completed':0,'initialization_failed':0,'initialization_unknown_inflight':0,
        'message_attempted':0,'message_completed':0,'message_failed':0,'message_unknown_inflight':0,
        'extra_attempted':0,'extra_completed':0,'extra_failed':0,'extra_unknown_inflight':0,
        'full_A_forwards':0,'net_encode_calls':0,'encoder_calls':0,'GRU_replays':0,'sampler_calls':0,'diffusion_calls':0}
    write_json(out/'STARTED.json',{'pid':os.getpid(),'argv':sys.argv,'time':time.time(),'caps':{'processes':4,'message_per_process':2,'message_total':8,'extra_T1_only':8}})
    resource_index=0
    def resource_gate(initial=False):
        nonlocal resource_index
        resource=gpu_snapshot();resource_index+=1
        write_json(out/f'resources/{resource_index:02d}.json',resource)
        require_idle(resource,uuid,allow_self=not initial,allow_desktop_graphics=True)
    try:
        resource_gate(True);os.environ['CUDA_VISIBLE_DEVICES']=uuid
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        for name in SOURCES:
            require(subprocess.check_output(['git','show',commit+':'+name])==(REPO/name).read_bytes(),'Uncommitted source','BLOCKED_INPUT')
        cpu=read(ROOT/'CPU_CHECKS.json')
        require(cpu['status']=='PASS' and all(sha256(REPO/p)==h for p,h in cpu['tested_sources_sha256'].items()),'CPU gate not authenticated','BLOCKED_INPUT')
        counts['initialization_attempted']=1;counts['initialization_unknown_inflight']=1;write_json(out/'INITIALIZING.json',counts)
        inputs,layer,info=authenticate();write_json(out/'INPUT_MANIFEST.json',info)
        ref=info['settings_reference']
        torch.backends.cudnn.deterministic=ref['cudnn_deterministic'];torch.backends.cudnn.benchmark=ref['cudnn_benchmark']
        layer=layer.to('cuda:0').eval();inputs={k:v.to('cuda:0') for k,v in inputs.items()}
        settings=execution_settings(torch)
        require(settings==ref,'Execution settings mismatch','BLOCKED_INPUT')
        inp_before={k:fingerprint(v) for k,v in inputs.items()};state_before=state_fingerprint(layer)
        require(all(all(v[f]==info['inputs'][k]['T0'][f] for f in v) for k,v in inp_before.items()),'GPU input contract','BLOCKED_INPUT')
        require(sys.gettrace() is None and sys.getprofile() is None,'Active profiler','BLOCKED_INPUT')
        require(all(not m.training for m in layer.modules()),'Non-eval module','BLOCKED_INPUT')
        saved_rng=torch.load(OLD/'T0/RNG.pt',weights_only=False,map_location='cpu')['before'];rng_restore(saved_rng)
        rng_before=rng_snapshot(True);require(_rng_equal(saved_rng,rng_before),'RNG restore failed','BLOCKED_INPUT')
        counts['initialization_completed']=1;counts['initialization_unknown_inflight']=0
        write_json(out/'INITIALIZED.json',{'source_commit':commit,'settings':settings,'counts':dict(counts),'desktop_allowlist':sorted(DESKTOP_EXECUTABLES),
            'RNG_source':'authenticated prior T0 batch before; not historical message-entry snapshot','precision':'original FP32 island; baseline cudnn flags restored, deterministic_algorithms not enabled'})
        resource_gate();observer=MessageObserver(layer);outputs=[]
        with torch.no_grad(),torch.autocast(device_type='cuda',enabled=False):
            for index in range(2):
                counts['message_attempted']+=1;counts['message_unknown_inflight']=1
                write_json(out/f'budget/{index+1}_attempt.json',dict(counts))
                with observer if process.startswith('T') and index==1 else contextlib.nullcontext():
                    value=layer(**inputs)
                outputs.append(value);counts['message_completed']+=1;counts['message_unknown_inflight']=0
                write_json(out/f'budget/{index+1}_complete.json',dict(counts))
        after_rng=rng_snapshot(True)
        validation={'input_unchanged':inp_before=={k:fingerprint(v) for k,v in inputs.items()},
            'model_unchanged':state_before==state_fingerprint(layer),'global_RNG_unchanged':_rng_equal(rng_before,after_rng),
            'hooks_restored':sys.gettrace() is None and sys.getprofile() is None and all(not m._forward_hooks and not m._forward_pre_hooks for m in layer.modules())}
        save_torch(out/'RNG.pt',{'before':rng_before,'after_message_batch':after_rng})
        common=Recorder()
        for index,value in enumerate(outputs,1):common.take(f'call{index}/h',value)
        tensor,meta=common.finish();meta['observer_cost']='All arms retain two normal returns until batch end; common ledger and batch-external input/model/RNG checks'
        save_torch(out/'OUTPUTS.pt',tensor);write_json(out/'OUTPUT_CONTRACTS.json',meta)
        trace=trace_meta=None
        if process.startswith('T'):
            trace,trace_meta=observer.finish()
            save_torch(out/'TRACE.pt',trace);write_json(out/'TRACE_CONTRACTS.json',trace_meta)
            validation['message_mlp_observed_calls']=observer.message_calls
        validation['snapshots_valid']=not meta['invalid_snapshots'] and (trace_meta is None or not trace_meta['invalid_snapshots'])
        validation['source_commit']=commit;validation['state']=state_before
        write_json(out/'VALIDATION.json',validation)
        require(all(validation[k] for k in ('input_unchanged','model_unchanged','global_RNG_unchanged','hooks_restored','snapshots_valid')),'Observation/RNG/state failure: stop','TRACE_INVALID')
        if process.startswith('T'):require(observer.message_calls==2,'Unexpected module count','TRACE_INVALID')
        resource_gate()
        extra={'status':'NOT_RUN','reason':'Only T1 may evaluate gate','attempted':0}
        if process=='T1':
            prior=ROOT/'T0'
            require((prior/'COMPLETE.json').is_file(),'T0 incomplete','BLOCKED_INPUT')
            other=torch.load(prior/'TRACE.pt',weights_only=True,map_location='cpu');other_meta=read(prior/'TRACE_CONTRACTS.json')
            previous=read(prior/'VALIDATION.json');previous_init=read(prior/'INITIALIZED.json')
            gate=compare_traces(other,trace,other_meta,trace_meta)
            gate['external_contract']=all(previous[k] for k in ('input_unchanged','model_unchanged','global_RNG_unchanged','hooks_restored','snapshots_valid')) and previous['state']==state_before and previous_init['settings']==settings and previous_init['source_commit']==commit
            gate['CPU_semantics_authenticated']=True
            write_json(out/'EXTRA_GATE.json',gate)
            if gate['gate_pass'] and gate['external_contract']:
                live=observer.recorder.tensors
                versions={k:v._version for k,v in live.items()}
                extra=controlled(layer,live,gate,out,counts,resource_gate)
                post=rng_snapshot(True)
                save_torch(out/'EXTRA_RNG.pt',{'before':after_rng,'after':post})
                extra_validation={'RNG_unchanged':_rng_equal(after_rng,post),'model_unchanged':state_before==state_fingerprint(layer),
                    'captured_inputs_versions_unchanged':versions=={k:v._version for k,v in live.items()},
                    'inputs_unchanged':inp_before=={k:fingerprint(v) for k,v in inputs.items()}}
                write_json(out/'EXTRA_VALIDATION.json',extra_validation)
                require(all(extra_validation.values()),'Extra diagnostic state changed','TRACE_INVALID')
            else:extra={'status':'NOT_RUN','attempted':0,'reason':gate['reason'] if not gate['gate_pass'] else 'External contract failed'}
        write_json(out/'EXTRA_STATUS.json',extra)
        resource_gate();write_json(out/'COMPLETE.json',{'status':'COMPLETE','source_commit':commit,**counts})
        print(json.dumps({'process':process,'extra_status':extra['status'],**counts}),flush=True)
    except BaseException as error:
        for kind in ('initialization','message','extra'):
            if counts[kind+'_unknown_inflight']:
                counts[kind+'_failed']+=1;counts[kind+'_unknown_inflight']=0
        write_json(out/'FAILED.json',{'status':getattr(error,'status','FAILED'),'error':repr(error),**counts})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--process',choices=['U0','U1','T0','T1'],required=True);parser.add_argument('--gpu-uuid',required=True)
    args=parser.parse_args();run(args.process,args.gpu_uuid)
