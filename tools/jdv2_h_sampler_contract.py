"""Authenticated fixed h pair / prepared inputs and scoped injection, no model replay."""
import json
import subprocess
from pathlib import Path
import torch
from tools.jdv2_stage_a_bank import REPO,CHECKPOINT,CHECKPOINT_SHA,CONFIG,CONFIG_SHA,sha256,require,code_provenance
from tools.jdv2_restart_observer import compare_tensor,Recorder
from tools.jdv2_social_local import fingerprint

BASE='4bf2f4ab79e891a96315d67f2d137bac66d2f850'
AUDITS=REPO/'outputs/joint_dependency_v2/eth/joint_dependency_v2/audits'
ROOT=AUDITS/'h_to_sampler_4bf2f4a'
OLD=AUDITS/'restart_trace_ea846fc/P2'
DOCS=REPO/'docs/joint_dependency_v2/reviews'
MESSAGE=DOCS/'2026-10-04_2156530_message_layer_local'
RESTART=DOCS/'2026-10-04_80ee0e2_restart_trace'
SOCIAL=DOCS/'2026-10-04_0ffdf7c_social_encoder_local'
SOURCES=['tools/jdv2_h_sampler_contract.py','tools/jdv2_h_sampler_probe.py','tools/jdv2_h_sampler_cpu.py','tools/jdv2_h_sampler_analyze.py']
H_HASH=['e61dbbf5d790db495fae8392ffde6caa7a95a25905032f88d5c8723565086a5a','6f63c893cb36dbbcc9753b2b30d4d373858281347caa9d6a3892afa16d6d2645']

def read(p):return json.loads(p.read_text())
def load(p):return torch.load(p,weights_only=True,map_location='cpu')

def authenticate():
    require(not subprocess.check_output(['git','diff',BASE,'--','src','configs']),'Production/config differs','BLOCKED_INPUT')
    info={'base_commit':BASE,'code':code_provenance(),'authenticated_files':{}}
    for directory in (MESSAGE,RESTART,SOCIAL):
        for name,h in read(directory/'ARTIFACT_HASHES.json')['files'].items():
            require(sha256(directory/name)==h,'Archive hash '+str(directory/name),'BLOCKED_INPUT')
    require(sha256(MESSAGE/'REVIEW.md')=='ae0d90881a3a554939f097a49d12b08cd6ed17ec3aad79a102a4dc36a92669ea','Report hash','BLOCKED_INPUT')
    require(sha256(MESSAGE/'RESULTS.json')=='6116ce42357cb9a31f4f5c0aff38be1257d46f1a40bef9d34f42bb0416627ac2','Results hash','BLOCKED_INPUT')
    def verify(root,manifest,names):
        for name in names:
            row=manifest[name];require(sha256(root/name)==row['sha256'],'Native hash '+name,'BLOCKED_INPUT')
            info['authenticated_files'][str((root/name).relative_to(REPO))]=row
    message_root=AUDITS/'message_layer_local_a13eae9'
    verify(message_root,read(MESSAGE/'ARTIFACT_MANIFEST.json')['files'],['U0/'+n for n in ('OUTPUTS.pt','OUTPUT_CONTRACTS.json','INPUT_MANIFEST.json','INITIALIZED.json','VALIDATION.json','COMPLETE.json')])
    verify(OLD.parent,read(RESTART/'TRACE_MANIFEST.json')['files'],['P2/'+n for n in ('TARGET.pt','TARGET_METADATA.json','TRACE.pt','TRACE_METADATA.json','STATE.json','RNG.pt','RESOLVED_ARGS.json','PROVENANCE.json','windows/13.json')])
    social_root=AUDITS/'social_encoder_local_634a467'
    verify(social_root,read(SOCIAL/'ARTIFACT_MANIFEST.json')['files'],['T0/'+n for n in ('TRACE.pt','TRACE_CONTRACTS.json','INPUT_MANIFEST.json')])
    h_raw=load(message_root/'U0/OUTPUTS.pt');hm=read(message_root/'U0/OUTPUT_CONTRACTS.json')
    hs={label:h_raw[f'call{i+1}/h'] for i,label in enumerate(('A','B'))}
    for i,(label,t) in enumerate(hs.items()):
        row=hm['tensors'][f'call{i+1}/h'];fp=fingerprint(t)
        require(row['valid_snapshot'] and all(fp[k]==row[k] for k in fp),'H snapshot/layout','BLOCKED_INPUT')
        require(fp['raw_sha256']==H_HASH[i] and list(t.shape)==[9,128] and t.dtype==torch.float32,'Predeclared H','BLOCKED_INPUT')
    delta=compare_tensor(hs['A'],hs['B']);require(delta['different_elements']==769 and delta['max_abs']==4.76837158203125e-7,'H pair changed','BLOCKED_INPUT')
    target=load(OLD/'TARGET.pt');tm=read(OLD/'TARGET_METADATA.json');trace=load(OLD/'TRACE.pt');trm=read(OLD/'TRACE_METADATA.json')
    flat={k:v for k,v in target.items() if k.startswith('inputs/')}
    for k,v in flat.items():
        row=tm['tensors'][k];fp=fingerprint(v)
        require(row['valid_snapshot'] and all(fp[f]==row[f] for f in fp),'Input requires unsupported layout reconstruction '+k,'BLOCKED_INPUT')
        key='GDTS._jdv2_goal_outputs/000/input/'+k
        require(trm['tensors'][key]['valid_snapshot'] and compare_tensor(v,trace[key])['bitwise_equal'],'Target/trace input mismatch','BLOCKED_INPUT')
    require(len({tm['tensors'][k]['alias_group'] for k in flat})==len(flat),'Input aliases need reconstruction','BLOCKED_INPUT')
    require(len({t.untyped_storage()._cdata for t in flat.values()})==len(flat),'Unexpected saved input aliases','BLOCKED_INPUT')
    inputs={};
    for k,v in flat.items():
        parts=k.split('/')[1:];node=inputs
        for name in parts[:-1]:node=node.setdefault(name,{})
        node[parts[-1]]=v
    inputs['jdv2_cache']['cache_id']=[tm['values']['inputs/jdv2_cache/cache_id/0']]
    require(inputs['jdv2_cache']['cache_id']==['valid-000013'],'Wrong window','BLOCKED_INPUT')
    sm=read(social_root/'T0/INPUT_MANIFEST.json');st=load(social_root/'T0/TRACE.pt');sc=read(social_root/'T0/TRACE_CONTRACTS.json')
    mi=read(message_root/'U0/INPUT_MANIFEST.json')
    for name,original in [('obs_traj',inputs['obs_traj_world']),('scene_index',inputs['scene_index']),('edge_index',inputs['jdv2_cache']['edge_index'].long()),('edge_feat',inputs['jdv2_cache']['edge_feat']),('edge_weight',inputs['jdv2_cache']['edge_weight'])]:
        require(fingerprint(original)['raw_sha256']==sm['inputs'][name]['P2']['raw_sha256'],'Window provenance '+name,'BLOCKED_INPUT')
        require(compare_tensor(original,trace['SocialMotionEncoder.forward/000/input/'+name])['bitwise_equal'],'Encoder source mismatch','BLOCKED_INPUT')
    for name in ('agent_feat','edge_index','edge_feat','edge_weight'):
        require(fingerprint(st['message_layer_0/input/'+name])['raw_sha256']==mi['inputs'][name]['T0']['raw_sha256'],'Message/encoder source chain','BLOCKED_INPUT')
    require(compare_tensor(st['message_layer_0/output'],st['encoder/final_h'])['bitwise_equal'],'Message is not final h','BLOCKED_INPUT')
    require(sc['tensors']['message_layer_0/output']['alias_group']==sc['tensors']['encoder/final_h']['alias_group'],'Final projection alias','BLOCKED_INPUT')
    require(sha256(REPO/CHECKPOINT)==CHECKPOINT_SHA and sha256(REPO/CONFIG)==CONFIG_SHA,'Frozen model hash','BLOCKED_INPUT')
    args=read(OLD/'RESOLVED_ARGS.json');require(args['social_attention_layers']==1 and args['amp_dtype']=='bf16' and args['jdv2_refinement_policy']=='exact_lexicographic_persistent_tie','Route args','BLOCKED_INPUT')
    info.update(h={k:fingerprint(v) for k,v in hs.items()},h_difference=delta,inputs={k:tm['tensors'][k] for k in flat},
        input_values={'jdv2_cache/cache_id':inputs['jdv2_cache']['cache_id']},checkpoint_sha256=CHECKPOINT_SHA,config_sha256=CONFIG_SHA,
        source_window={'seed':2036,'window':13,'cache_id':'valid-000013','N':9,'E':36,'K':21,'P':20},
        provenance_chain='P2 prepared inputs = P2 encoder inputs = social T0 input; social T0 message input = message U0 input. Single layer output aliases final encoder h.',
        input_layout='22 distinct storage groups, original strides/offsets including noncontiguous world_coord retained; CPU .to(cuda) must revalidate. No omitted non-tensor scene field is read by explicit-cache method.',
        settings_reference=read(OLD/'STATE.json')['settings'],
        context={'seed':2036,'window_index':13,'reset_before_each_call':True},
        precision='original outer CUDA/BF16 plus existing FP32 islands; injected h FP32',
        limitations=['Historical CUDA address/allocator/stream not restored','Same observer on all arms does not certify neutrality','Labels only used by original post-decision logs; teacher disabled'])
    return hs,inputs,info


class InjectH:
    def __init__(self,encoder,h):self.encoder=encoder;self.h=h;self.calls=0
    def __enter__(self):
        self.had='forward' in self.encoder.__dict__;self.old=self.encoder.__dict__.get('forward')
        def injected(*args,**kwargs):
            self.calls+=1
            if self.calls!=1:raise RuntimeError('Injected encoder called more than once')
            return self.h
        self.encoder.forward=injected
        return self
    def __exit__(self,*exc):
        if self.had:self.encoder.forward=self.old
        else:del self.encoder.forward
        return False


def input_fingerprint(inputs):
    r=Recorder();r.take('inputs',inputs)
    return {k:{**fingerprint(t),'alias_group':r.metadata[k]['alias_group'],'version':t._version} for k,t in r.tensors.items()}
