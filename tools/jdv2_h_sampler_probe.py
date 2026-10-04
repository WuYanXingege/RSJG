#!/usr/bin/env python3
"""Two exclusive four-call h replacement slots; original goal/sampler arithmetic."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tools.jdv2_h_sampler_contract import *
from tools.jdv2_restart_observer import BoundaryObserver,state_fingerprint
from tools.jdv2_restart_probe import save_torch,execution_settings
from tools.jdv2_stage_a_bank import write_json


class Observer(BoundaryObserver):
    """Reuse audited boundaries but never wrap random calls; common callback on all arms."""
    def __init__(self,net=None,counts=None,ledger=None):
        super().__init__(net,enabled=True)
        from src.models.joint_dependency_v2 import exact_lexicographic_assignment as exact
        self.codes[exact.stable_tie_seed.__code__]='stable_tie_seed'
        self.net=net;self.budget=counts;self.ledger=ledger;self.forbidden={}
        if net is not None:
            from src.models.social_encoder import SocialMotionEncoder,_SocialMessageLayer
            for label,fn in [('full_A',net.forward),('net_encode',net.encode),('encoder',SocialMotionEncoder.forward),
                ('message',_SocialMessageLayer.forward),('GRU',torch.nn.GRU.forward),('history_encoder',net.encoder.encode_hist),
                ('diffusion',net.ts_sample),('contexts',net._jdv2_contexts),('jdv2_encode',net._jdv2_encode),('heatmap',net.goal_module.forward)]:
                self.forbidden[fn.__code__]=label

    def profile(self,frame,event,arg):
        if event=='call' and frame.f_code in self.forbidden:
            label=self.forbidden[frame.f_code]
            self.budget['forbidden_'+label]+=1
            raise RuntimeError('Forbidden original forward: '+label)
        label=self.codes.get(frame.f_code)
        if label=='ParallelConditionalSampler.forward' and self.budget is not None:
            if event=='call':
                require(self.budget['sampler_attempted']<4,'Sampler budget')
                self.budget['sampler_attempted']+=1;self.budget['sampler_unknown_inflight']=1
                write_json(self.ledger/f'sampler_{self.budget["sampler_attempted"]}_attempt.json',self.budget)
            elif event=='return' and arg is not None:
                self.budget['sampler_completed']+=1;self.budget['sampler_unknown_inflight']=0
                write_json(self.ledger/f'sampler_{self.budget["sampler_attempted"]}_complete.json',self.budget)
        return super().profile(frame,event,arg)

    def callback(self,**payload):
        self.recorder.take('callback/round'+str(payload['round_index']),payload)

    def __enter__(self):
        require(sys.getprofile() is None and sys.gettrace() is None,'Existing observation state')
        if self.net is not None:
            self.old_callback=self.net.jdv2_sampler.diagnostic_callback
            require(self.old_callback is None,'Existing diagnostic callback')
            self.net.jdv2_sampler.diagnostic_callback=self.callback
        sys.setprofile(self.profile)
        return self

    def __exit__(self,*exc):
        sys.setprofile(None)
        if self.net is not None:self.net.jdv2_sampler.diagnostic_callback=self.old_callback
        return False


def all_buffers(net):return {k:{**fingerprint(v),'version':v._version} for k,v in net.named_buffers()}


def run(process,uuid):
    from tools.jdv2_export_stage_a_bank import gpu_snapshot,require_idle
    from tools.audit_jdv2_stage_a import _active_evaluator
    from tools.jdv2_stage_b_v1_failure_mechanism_audit import rng_snapshot,rng_restore
    from tools.jdv2_stage_b_identity_contract_audit import _rng_equal
    out=ROOT/process;out.mkdir(parents=True,exist_ok=False)
    counts={'processes_started':1,**{f'{kind}_{s}':0 for kind in ('initialization','fragment','sampler') for s in ('attempted','completed','failed','unknown_inflight')},
        **{'forbidden_'+k:0 for k in ('full_A','net_encode','encoder','message','GRU','history_encoder','diffusion','contexts','jdv2_encode','heatmap')},'extra_GPU_replays':0}
    write_json(out/'STARTED.json',{'pid':os.getpid(),'time':time.time(),'argv':sys.argv,'sequence':'ABAB' if process=='P0' else 'BABA','caps':{'processes':2,'fragments':8,'sampler':8}})
    resource_index=0
    def resource(initial=False):
        nonlocal resource_index
        row=gpu_snapshot();resource_index+=1;write_json(out/f'resources/{resource_index:02d}.json',row)
        require_idle(row,uuid,allow_self=not initial,allow_desktop_graphics=True)
    try:
        resource(True);os.environ['CUDA_VISIBLE_DEVICES']=uuid
        if process=='P1':require((ROOT/'P0/COMPLETE.json').is_file(),'P0 incomplete; no automatic continuation')
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        for p in SOURCES:require(subprocess.check_output(['git','show',commit+':'+p])==(REPO/p).read_bytes(),'Uncommitted source','BLOCKED_INPUT')
        cpu=read(ROOT/'CPU_CHECKS.json');require(cpu['status']=='PASS' and all(sha256(REPO/p)==h for p,h in cpu['tested_sources_sha256'].items()),'CPU gate','BLOCKED_INPUT')
        counts['initialization_attempted']=1;counts['initialization_unknown_inflight']=1;write_json(out/'INITIALIZING.json',counts)
        hs,inputs,info=authenticate();write_json(out/'INPUT_MANIFEST.json',info)
        ref=info['settings_reference'];torch.backends.cudnn.deterministic=ref['cudnn_deterministic'];torch.backends.cudnn.benchmark=ref['cudnn_benchmark']
        evaluator,epoch=_active_evaluator(str(REPO/CONFIG),str(REPO/CHECKPOINT),out/'runtime','cuda:0');net=evaluator.net
        require(epoch==13 and net.strict_no_z and not net.training and evaluator.amp_enabled and evaluator.amp_dtype==torch.bfloat16,'Canonical route','BLOCKED_INPUT')
        require(len(net.social_encoder.message_layers)==1 and isinstance(net.social_encoder.output_projection,torch.nn.Identity),'H equivalence','BLOCKED_INPUT')
        require(all(not m.training for m in net.modules()),'Not eval','BLOCKED_INPUT')
        clean=lambda d:{k:v for k,v in d.items() if k not in ('model_dir','save_dir')}
        require(clean(vars(net.args))==clean(read(OLD/'RESOLVED_ARGS.json')),'Resolved args','BLOCKED_INPUT')
        head=net.jdv2_corrector.output[-1]
        require(torch.count_nonzero(head.weight).item()==0 and torch.count_nonzero(head.bias).item()==0,'Corrector not strict zero','BLOCKED_INPUT')
        state=state_fingerprint(net);historical=read(OLD/'STATE.json')['before']
        require(set(state)==set(historical) and all(all(v[f]==historical[k][f] for f in ('dtype','shape','sha256')) for k,v in state.items()),'Strict full state mismatch','BLOCKED_INPUT')
        buffers=all_buffers(net)
        def move(x):
            if torch.is_tensor(x):return x.to('cuda:0')
            if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
            return x
        inputs=move(inputs);hs=move(hs);h=torch.empty_like(hs['A']);h_storage=h.untyped_storage()._cdata
        before_inputs=input_fingerprint(inputs)
        for k,v in before_inputs.items():
            require(all(v[f]==info['inputs'][k][f] for f in ('dtype','shape','stride','storage_offset','raw_sha256')),'GPU input contract '+k,'BLOCKED_INPUT')
        require(len({v['alias_group'] for v in before_inputs.values()})==22,'GPU alias contract','BLOCKED_INPUT')
        settings=execution_settings(torch);require(settings==ref,'Settings mismatch','BLOCKED_INPUT')
        rng_restore(torch.load(OLD/'RNG.pt',weights_only=False,map_location='cpu')['before13'])
        initial_rng=rng_snapshot(True);write_json(out/'STATE.json',{'state':state,'all_buffers':buffers,'inputs':before_inputs})
        original_trace=load(OLD/'TRACE.pt');original_values=read(OLD/'TRACE_METADATA.json')['values']
        counts['initialization_completed']=1;counts['initialization_unknown_inflight']=0
        write_json(out/'INITIALIZED.json',{'source_commit':commit,'settings':settings,'counts':dict(counts),'args':vars(net.args),
            'preparation':'one fixed FP32 h buffer + two source h transfers; one copy_ before each call; all inputs transferred once; historical layouts checked',
            'RNG_source':'P2 before13, not historical sampler-entry state; only initial local generator consumed by fragment',
            'strict_load':True,'h_projection_identity':True,'corrector_strict_zero':True})
        for index,label in enumerate('ABAB' if process=='P0' else 'BABA',1):
            resource();folder=out/f'call{index}_{label}'
            h.copy_(hs[label]);h_before=fingerprint(h);version=h._version
            require(h.untyped_storage()._cdata==h_storage and h_before==info['h'][label],'H workbuffer','CONTRACT_FAILED')
            net.jdv2_sampler.set_sampling_context(2036,13);context_before=net.jdv2_sampler._sampling_context
            rng_before=rng_snapshot(True);require(_rng_equal(initial_rng,rng_before),'Unexpected between-call RNG change','CONTRACT_FAILED')
            counts['fragment_attempted']+=1;counts['fragment_unknown_inflight']=1;write_json(out/f'budget/fragment_{index}_attempt.json',counts)
            observer=Observer(net,counts,out/'budget');original_forward=net.social_encoder.forward
            with InjectH(net.social_encoder,h) as injection,observer,torch.no_grad():
                with evaluator._autocast_context():
                    result=net._jdv2_goal_outputs(inputs,None,sample=True,include_teacher=False)
                require(result['agent_feat'] is h and injection.calls==1,'H injection return contract','CONTRACT_FAILED')
                observer.recorder.take('injected_h',h)
                payload,meta=observer.recorder.finish()  # before next copy_, still in injection/observer scope
            counts['fragment_completed']+=1;counts['fragment_unknown_inflight']=0;write_json(out/f'budget/fragment_{index}_complete.json',counts)
            meta['observer_cost']='Same limited profile/callback all arms; no random wrappers; held references; CPU ledger at sampler boundary; per-call native dump before next h copy_; generator.get_state; no extra hot tensor arithmetic'
            save_torch(folder/'TRACE.pt',payload);write_json(folder/'TRACE_METADATA.json',meta)
            rng_after=rng_snapshot(True);save_torch(folder/'RNG.pt',{'before':rng_before,'after':rng_after})
            random_keys=[k for k in original_trace if k.startswith('explicit_generators/') or k.startswith('_unit_gumbel/')]
            random_equal={k:compare_tensor(payload[k],original_trace[k])['bitwise_equal'] for k in random_keys}
            priority_keys=[k for k in meta['values'] if k.startswith('solve_exact_persistent_tie/') and '/output/priorities/' in k]
            priorities_equal=bool(priority_keys) and all(meta['values'][k]==original_values[k] for k in priority_keys)
            validation={'input_unchanged':input_fingerprint(inputs)==before_inputs,'model_unchanged':state_fingerprint(net)==state and all_buffers(net)==buffers,
                'global_RNG_unchanged':_rng_equal(rng_before,rng_after),'h_unchanged_inside_fragment':version==h._version and fingerprint(h)==h_before,
                'same_h_storage':h.untyped_storage()._cdata==h_storage,'snapshot_valid':not meta['invalid_snapshots'] and meta['native_dtype_roundtrip'],
                'scope_restored':net.social_encoder.forward==original_forward and sys.getprofile() is None and net.jdv2_sampler.diagnostic_callback is None,
                'sampling_context_consumed':context_before==(2036,13) and net.jdv2_sampler._sampling_context is None,
                'actual_random_payload_equal_P2':all(random_equal.values()),'actual_priorities_equal_P2':priorities_equal,
                'single_sampler':observer.counts['ParallelConditionalSampler.forward']==1,'single_gumbel':observer.counts['_unit_gumbel']==1,
                'exact_problems_18':observer.counts['solve_exact_persistent_tie']==18}
            write_json(folder/'VALIDATION.json',{'checks':validation,'actual_random_checks':random_equal,'h_label':label,'h_version':version,
                'h_storage_id_within_process':str(h_storage),'counts':dict(observer.counts),'context_before':context_before,'context_after':net.jdv2_sampler._sampling_context,
                'post_decision_original_logs':net.last_joint_diagnostics})
            require(all(validation.values()),'Per-call contract failed: '+repr(validation),'CONTRACT_FAILED')
            print(json.dumps({'process':process,'call':index,'h':label,'status':'COMPLETE_VALIDATED',**counts}),flush=True)
            del result,observer,payload,meta,injection
        resource();write_json(out/'COMPLETE.json',{'status':'COMPLETE','source_commit':commit,**counts})
    except BaseException as error:
        for kind in ('initialization','fragment','sampler'):
            if counts[kind+'_unknown_inflight']:counts[kind+'_failed']+=1;counts[kind+'_unknown_inflight']=0
        write_json(out/'FAILED.json',{'status':getattr(error,'status','FAILED'),'error':repr(error),**counts})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--process',choices=['P0','P1'],required=True);parser.add_argument('--gpu-uuid',required=True)
    args=parser.parse_args();run(args.process,args.gpu_uuid)
