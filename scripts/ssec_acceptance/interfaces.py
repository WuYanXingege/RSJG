"""Native callbacks, zero-update writer, keyed DDIM and read-only official bank checks."""
from common import *
from bcep_scene_energy.model import build
from bcep_v3.fixtures import SyntheticData,SyntheticBank
from bcep_v2.stages import joint_callbacks
from bcep_v2.state import run_stage


def main():
    setup(cuda=True);seed();c=config();m,origin=build(c);m.cuda().train();before=state_hash(m)
    data=SyntheticData((2,9),pixels=256);bank=SyntheticBank()
    backward,validate=joint_callbacks(m,data,'cuda',c,bank,bank)
    metrics=backward([0,1],torch.Generator().manual_seed(3201),torch.Generator().manual_seed(3202))
    callback_grad={prefix:gradnorm([p.grad for name,p in m.named_parameters() if name.startswith(prefix)]) for prefix in ['coupling.energy.phi.','coupling.energy.rho.','bridge.encoder.']}
    assert all(v>0 for v in callback_grad.values());m.zero_grad(set_to_none=True)
    # No epoch can complete: the ORIGINAL controller exits before any backward/step.
    attempt=1;folder=OUT/'zero_update_state_fixture'
    while folder.exists():
        attempt+=1;folder=OUT/f'zero_update_state_fixture_attempt_{attempt:02d}'
    recipe=dict(seed=3201,lr=1e-4,weight_decay=1e-4,clip=5.,logical_scenes=16,max_epochs=100,minimum=30,patience=15,scheduler_horizon=100)
    def forbidden(*args):raise AssertionError('zero-update writer must not invoke a training callback')
    state=run_stage(folder,dict(stage='joint',fold='SYNTHETIC_INTERFACE_ONLY'),recipe,m,[0,1],forbidden,forbidden,0,resume=False)
    saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
    assert state['update']==state['completed_epoch']==0 and state['status']=='SESSION_BUDGET_PAUSED'
    required=['model','optimizer','scheduler','rng','cursor','order','history','completed_epoch','update','trainable','identity','recipe']
    assert all(k in saved for k in required)
    seed();restored,_=build(c);restored.load_state_dict(saved['model'],strict=True)
    from bcep_v2.state import optimizer
    opt=optimizer(restored,recipe['lr'],recipe['weight_decay']);opt.load_state_dict(saved['optimizer'])
    assert state_hash(restored)==before and all(name in saved['trainable'] for name,p in restored.named_parameters())
    assert len(list(folder.glob('epoch_*.pt')))==0 and not (folder/'TOP5.json').exists()
    # Original inference calls coupling once, then keeps candidates/noise fixed for DDIM.
    import bcep_joint.inference as inference
    x=data.batch([0],'selection_dev').to('cuda');g=bank.batch([0],data.batch([0],'selection_dev')).cuda()
    assign=m.coupling.assign;sample=inference.differentiable_sample;assign_calls=[0];samples=[]
    def assigned(*a,**kw):assign_calls[0]+=1;return assign(*a,**kw)
    def sampled(*a,**kw):
        samples.append(dict(steps=a[3],noise=a[2].detach().cpu(),goals=kw['proposal_goals'].detach().cpu()))
        return sample(*a,**kw)
    m.coupling.assign=assigned;inference.differentiable_sample=sampled
    try:_,paths,pi,checks=inference.generate(m,x.observed,x.semantic_maps,x.valid,x.scene_ids,x.agent_ids,g,2035,mode='H')
    finally:m.coupling.assign=assign;inference.differentiable_sample=sample
    expected=inference.keyed_noise(x.scene_ids,x.agent_ids,x.valid,2035)
    assert assign_calls[0]==1 and len(samples)==1 and samples[0]['steps']==20
    assert torch.equal(samples[0]['noise'],inference.gather_candidates(expected,pi.cpu()))
    assert torch.equal(samples[0]['goals'],inference.gather_candidates(g,pi).cpu())
    assert state_hash(m)==before
    # Verify official identity; only selection_dev FORWARD probes. No official backward.
    from bcep_v3.formal_inputs import FormalInputs
    inputs=FormalInputs();seed();real,_=build(inputs.config);real.cuda().eval();real_before=state_hash(real)
    selected=[]
    for n in [1,2,9,42]:
        selected.append(next(i for i in inputs.split['selection_dev_indices'] if len(inputs.split['members'][i]['agent_ids'])==n))
    probes=[]
    with torch.no_grad():
        for index in selected:
            batch=inputs.data.batch([index],'selection_dev');goals=inputs.banks[1].batch([index],batch);x=batch.to('cuda');g=goals.cuda()
            context=real.encode_context(x.observed,x.semantic_maps,x.valid)
            h,pi,s=real.coupling.assign(context,g,real.bridge.representation.goal_scale,straight_through=False)
            e0=(g-x.future[:,None,:,-1]).norm(dim=-1).min(1).values
            e1=(h-x.future[:,None,:,-1]).norm(dim=-1).min(1).values
            assert torch.equal(e0,e1) and torch.equal(h,inference.gather_candidates(g,pi))
            sd=s.double();effective=sd-sd.mean(-1,keepdim=True)-sd.mean(-2,keepdim=True)+sd.mean((-1,-2),keepdim=True)
            probes.append(dict(index=index,N=int(x.valid.sum()),bank_candidate_sha256=hashlib.sha256(goals.contiguous().numpy().tobytes()).hexdigest(),
                raw_score_RMS=float(sd.square().mean().sqrt()),effective_score_RMS=float(effective.square().mean().sqrt()),
                minFDE_max_abs_difference=float((e0-e1).abs().max()),exact_candidate_gather=True,
                accepted=sum(t['accepted'] for t in real.coupling.last_diagnostics[0]['trace']),optimizer_updates=0))
    assert state_hash(real)==real_before
    csvwrite(DOC/'OFFICIAL_DEV_FORWARD_PROBES.csv',probes)
    # Historical weights: strict compatibility loading ONLY, never new initialization.
    from bcep_v2.joint import build_joint
    from bcep_v3.model import build as v3build
    refs=[('v2_e5',OUT.parent/'bcep_ebjd_v2_integrated/zara2/joint/epoch_005.pt'),
          ('v2_e26',OUT.parent/'bcep_ebjd_v2_integrated/zara2/joint/epoch_026.pt'),
          ('v3_e12',OUT.parent/'zara2_bcep_v3_stage2_20261120/joint/epoch_012.pt')]
    compatibility=[]
    for label,path in refs:
        prior=torch.load(path,map_location='cpu',weights_only=False)
        old=build_joint(inputs.config) if label.startswith('v2') else v3build(inputs.config)[0]
        old.load_state_dict(prior['model'],strict=True)
        assert all(torch.equal(old.state_dict()[k],v) for k,v in prior['model'].items())
        compatibility.append(dict(label=label,sha256=sha(path),strict_load=True,used_as_ssec_initialization=False))
        del old,prior
    assert GUARD['optimizer_step_calls']==0
    write(DOC/'INTERFACE_RECEIPT.json',dict(passed=True,optimizer_guard=GUARD,real_updates=0,synthetic_updates=0,
        native_grouped_callback_gradient_norms=callback_grad,callback_valid_scene_count=metrics['valid_scene_count'],
        zero_update_native_state_status=state['status'],zero_update_state_keys=required,strict_model_optimizer_state_reload=True,
        scheduler_steps_executed=0,completed_epochs=0,new_top5_created=False,
        native_H_JADE_selection_controller_unchanged=True,top5_completed_epoch_behavior_not_reexecuted_due_to_zero_update_prohibition=True,
        native_inference_checks=checks,coupling_calls_during_20step_DDIM=assign_calls[0],keyed_noise_exact=True,
        official_probe_purpose='selection_dev forward only; no gradients, no model selection, no test data',
        official_counts=inputs.counts,provider=inputs.provider,bank_sha256=[b.hash for b in inputs.banks],
        historical_compatibility=compatibility,model_states_unchanged=True))
    print('Native grouped gradients, zero-update state, DDIM/noise and official dev forward probes PASS',flush=True)


if __name__=='__main__':main()
