"""Official train/dev identity and real-data backward, with no optimizer steps."""
from common import *
from bcep_v3.formal_inputs import FormalInputs
from bcep_scene_energy.model import build
from bcep_scene_energy.formal_callbacks import formal_callbacks
from bcep_scene_energy.formal_monitor import StepObserver,gradient_statistics
from bcep_scene_energy.formal_validation import window
from bcep_scene_energy.formal_contract import freeze,locks,source_manifest
from bcep_scene_energy.identity import bank_occurrence_ids
from bcep_v2.stages import joint_callbacks
from bcep_joint.inference import keyed_noise
from ebjd.unified import gather_candidates


def main():
    setup(cuda=True)
    if not (OUT/'PROTECTED_BEFORE.json').exists():write(OUT/'PROTECTED_BEFORE.json',protected())
    with locks(ROOTS):
        inputs=FormalInputs();frozen,origin=freeze(inputs)
        records=[]
        for purpose,bank in zip(['train','selection_dev'],inputs.banks):
            for index in inputs.split[purpose+'_indices']:
                row=bank.native.rows[index];record=inputs.split['records'][index]
                actual=digest(bank.native.folder/f'{index}.npy');assert actual==row['sha256']
                actual_record=digest(record['path']);assert actual_record==record['sha256']
                records.append(dict(purpose=purpose,index=index,agents=len(row['member']['agent_ids']),
                    member_sha256=canonical_hash(row['member']),candidate_expected_sha256=row['sha256'],candidate_actual_sha256=actual,
                    data_expected_sha256=record['sha256'],data_actual_sha256=actual_record))
        csvwrite(PUBLIC/'INPUT_ROWS.csv',records)
        write(PUBLIC/'REAL_INPUTS.json',dict(passed=True,counts=inputs.counts,provider=inputs.provider,
            banks=frozen['bank_sha256'],split_sha256=frozen['split_sha256'],all_rows_verified=len(records),
            provider_loaded_for_training=False,provider_in_optimizer=False,optimizer_guard=GUARD))
        seed_all(3201);model,_=build(inputs.config,variant='energy_only');initial=model_hash(model);model.cuda().train()
        assert initial==frozen['initial_model_sha256']
        parent=read(SOURCE/'docs/research/bcep_scene_energy/candidate_identity_20261121/GPU_RECEIPT.json')
        assert initial==parent['state_sha256']
        selected=[next(i for i in inputs.split['train_indices'] if len(inputs.split['members'][i]['agent_ids'])==n) for n in [1,2,9,57]]
        observer=StepObserver(model,OUT/'UNCOMMITTED_AUDIT_STEPS.jsonl')
        bw=formal_callbacks(model,inputs,'cuda',observer);rows=[]
        for index,n in zip(selected,[1,2,9,57]):
            seed_all(3201);model.zero_grad(set_to_none=True);model.train()
            start=time.perf_counter();metrics=bw([index],torch.Generator().manual_seed(3201),torch.Generator().manual_seed(3202));torch.cuda.synchronize()
            grad=observer.grad
            assert all(np.isfinite(v['gradient_l2']) for v in grad.values())
            if n>1:assert all(v['gradient_l2']>0 for v in grad.values())
            else:assert grad['bridge.encoder.']['gradient_l2']>0
            assert model_hash(model)==initial
            for prefix,value in grad.items():rows.append(dict(index=index,N=n,module=prefix,**value,parameter_delta_l2=0.,
                seconds=time.perf_counter()-start,allocated_peak_bytes=observer.peak_bytes,optimizer_updates=0))
            print('Official backward',n,'PASS; no step',flush=True)
        csvwrite(PUBLIC/'REAL_BACKWARD.csv',rows)
        # Actual native 16-scene grouped graph including all requested capacities.
        logical=list(dict.fromkeys(selected+inputs.split['train_indices'][:16]))[:16]
        model.zero_grad(set_to_none=True);seed_all(3201)
        metrics=bw(logical,torch.Generator().manual_seed(3201),torch.Generator().manual_seed(3202))
        assert metrics['valid_scene_count']==16 and model_hash(model)==initial
        write(PUBLIC/'LOGICAL_GROUP_BACKWARD.json',dict(passed=True,indices=logical,valid_scene_count=16,
            gradients=observer.grad,physical_groups=observer.groups,seconds=observer.seconds,peak_bytes=observer.peak_bytes,
            parameter_updates=0,optimizer_guard=GUARD))
        # The new identity/monitor view must preserve exact native backward.
        gradients=[];losses=[]
        plain,_=joint_callbacks(model,inputs.data,'cuda',inputs.config,*inputs.banks)
        for callback in [plain,bw]:
            model.zero_grad(set_to_none=True);model.train();seed_all(3201)
            metrics=callback([selected[1],selected[2]],torch.Generator().manual_seed(3201),torch.Generator().manual_seed(3202))
            gradients.append({n:p.grad.detach().cpu().clone() if p.grad is not None else None for n,p in model.named_parameters()});losses.append(metrics['loss'])
        assert losses[0]==losses[1]
        assert all((a is None and gradients[1][n] is None) or (a is not None and torch.equal(a,gradients[1][n])) for n,a in gradients[0].items())
        del gradients;model.zero_grad(set_to_none=True)
        # Capture ACTUAL native sampler inputs, not merely regenerate expected noise.
        import bcep_joint.inference as inference
        original=inference.differentiable_sample;captures=[]
        def sample(*a,**kw):
            captures.append(dict(steps=a[3],noise=a[2].detach().cpu(),goals=kw['proposal_goals'].detach().cpu()))
            return original(*a,**kw)
        checks=[]
        for index,n in zip(selected,[1,2,9,57]):
            batch=inputs.data.batch([index],'train');g,ids,_=bank_occurrence_ids(inputs.banks[0],[index],batch);captures.clear()
            inference.differentiable_sample=sample
            try:row,stats=window(model,batch,g,ids)
            finally:inference.differentiable_sample=original
            assert len(captures)==1 and captures[0]['steps']==20
            # Recover each exact original occurrence from distinct fixed-bank coordinates.
            h=captures[0]['goals'];match=(h[:,:,:,None,:]==g.permute(0,2,1,3)[:,None]).all(-1)
            assert (match.sum(-1)==1).all();pi=match.long().argmax(-1)
            z=keyed_noise(batch.scene_ids,batch.agent_ids,batch.valid,2035)
            assert torch.equal(captures[0]['noise'],gather_candidates(z,pi))
            checks.append(dict(index=index,N=n,actual_DDIM_steps=20,actual_sampler_calls=1,exact_endpoint=row['exact_endpoint'],actual_keyed_noise_exact=True))
        assert model_hash(model)==initial and not observer.path.exists() and observer.update==0
        write(PUBLIC/'REAL_ACCEPTANCE.json',dict(passed=True,initial_model_sha256=initial,previous_initial_fingerprint_exact=True,
            parameters=sum(p.numel() for p in model.parameters()),energy_only=model.coupling.csee is None,
            native_vs_instrumented_backward_exact=True,losses=losses,endpoint_noise_checks=checks,
            model_unchanged=True,actual_optimizer_step_hooks_fired=0,real_optimizer_updates=0,synthetic_optimizer_updates=0,
            implementation_hash=frozen['implementation_hash'],optimizer_guard=GUARD))
        print('Real-input identity, backward and sampler identity PASS',flush=True)
if __name__=='__main__':main()
