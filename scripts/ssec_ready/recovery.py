"""Enter the ORIGINAL resume path, pause before backward; never optimizer.step."""
import copy,random
from common import *
from bcep_v3.formal_inputs import FormalInputs
from bcep_scene_energy.model import build
from bcep_scene_energy.formal_contract import freeze,locks
from bcep_v2 import state as native
from bcep_joint.model import set_lr
from bcep_joint.common import save
from bcep_joint.checks import equal
from bcep_scene_energy.formal_monitor import StepObserver


def main():
    setup(cuda=True)
    with locks(ROOTS):
        inputs=FormalInputs();frozen,_=freeze(inputs);seed_all(3201);model,_=build(inputs.config);model.cuda()
        initial=model_hash(model);recipe=inputs.recipe;train_ids=inputs.split['train_indices']
        folder=OUT/'recovery_zero_update';attempt=1
        while folder.exists():attempt+=1;folder=OUT/f'recovery_zero_update_attempt_{attempt:02d}'
        identity=dict(frozen['identity'],experiment='SSEC_ZERO_UPDATE_RECOVERY_FIXTURE')
        def forbidden(*a,**kw):raise AssertionError('recovery audit must pause before backward/validation')
        observer=StepObserver(model,folder/'NO_STEPS.jsonl')
        with observer.native_optimizer_hook():
            state=native.run_stage(folder,identity,recipe,model,train_ids,forbidden,forbidden,0,resume=False)
        assert state['update']==state['completed_epoch']==0 and observer.update==0
        original=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        save(original,folder/'initial_zero_state.pt')
        # Build a valid pre-first-update state. Nonempty AdamW zero moments are
        # EXPLICITLY MATERIALIZED FIXTURE data, not results of an optimizer step.
        opt=native.optimizer(model,recipe['lr'],recipe['weight_decay'])
        for group in opt.param_groups:
            for p in group['params']:
                opt.state[p]=dict(step=torch.tensor(0.),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
        scheduler=set_lr(opt,1,100)
        order=torch.Generator().manual_seed(3201);pending=[train_ids[i] for i in torch.randperm(len(train_ids),generator=order).tolist()]
        saved=copy.deepcopy(original);saved.update(order=pending,cursor=0,scheduler=scheduler,optimizer=opt.state_dict(),
            status='ZERO_UPDATE_RECOVERY_FIXTURE',fixture='pending order and materialized zero AdamW slots; no parameter updates')
        saved['rng']['order']=order.get_state();save(saved,folder/'last.pt');save(saved,folder/'before_resume.pt')
        expected=torch.load(folder/'before_resume.pt',map_location='cpu',weights_only=False)
        # A fresh instance from the advanced SAME seed3201 RNG stream proves
        # that strict loading actually restores parameters, not just counters.
        recovered,_=build(inputs.config);recovered.cuda();different=model_hash(recovered)!=initial
        assert different
        random.random();np.random.rand();torch.rand(3);torch.rand(3,device='cuda')
        original_pause=native.pause_requested;original_factory=native.optimizer;captured=[];load_calls=[]
        original_load=recovered.load_state_dict
        def loaded(*a,**kw):load_calls.append(kw.get('strict'));return original_load(*a,**kw)
        recovered.load_state_dict=loaded
        def factory(*a,**kw):
            opt=original_factory(*a,**kw);captured.append(opt);return opt
        native.optimizer=factory;native.pause_requested=lambda:True
        try:
            result=native.run_stage(folder,identity,recipe,recovered,train_ids,forbidden,forbidden,1,resume=True)
        finally:
            native.optimizer=original_factory;native.pause_requested=original_pause;recovered.load_state_dict=original_load
        assert load_calls==[True] and len(captured)==1 and model_hash(recovered)==initial
        after=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        fields=['model','optimizer','scheduler','rng','cursor','order','completed_epoch','update','history','trainable',
                'epoch_sums','epoch_raw_sums','epoch_counts','epoch_steps','identity','recipe','max_updates']
        for k in fields:equal(expected[k],after[k])
        assert after['status']=='SESSION_BUDGET_PAUSED' and after['update']==after['completed_epoch']==0
        assert not (folder/'TOP5.json').exists() and not list(folder.glob('epoch_*.pt'))
        # Next data/RNG draws agree, not merely serialized presence of keys.
        cpu_gen=torch.Generator();cpu_gen.set_state(after['rng']['cpu'])
        assert torch.equal(torch.rand(5),torch.rand(5,generator=cpu_gen))
        gpu_gen=torch.Generator(device='cuda');gpu_gen.set_state(after['rng']['cuda'][0])
        assert torch.equal(torch.rand(5,device='cuda'),torch.rand(5,device='cuda',generator=gpu_gen))
        py=random.Random();py.setstate(after['rng']['python']);assert random.random()==py.random()
        npgen=np.random.RandomState();npgen.set_state(after['rng']['numpy']);assert np.random.rand()==npgen.rand()
        for key in ['order','view']:
            a=torch.Generator();b=torch.Generator();a.set_state(expected['rng'][key]);b.set_state(after['rng'][key])
            assert torch.equal(torch.randperm(17,generator=a),torch.randperm(17,generator=b))
        # Recipe mismatch must reject even if a pause would avoid all updates.
        native.pause_requested=lambda:True
        try:
            try:native.run_stage(folder,identity,dict(recipe,patience=16),recovered,train_ids,forbidden,forbidden,1,resume=True)
            except ValueError as error:assert 'recipe' in str(error)
            else:raise AssertionError('modified recovery recipe accepted')
        finally:native.pause_requested=original_pause
        write(PUBLIC/'RECOVERY_RECEIPT.json',dict(passed=True,optimizer_guard=GUARD,real_optimizer_updates=0,synthetic_optimizer_updates=0,
            original_native_resume_branch_executed=True,strict_model_load_calls=load_calls,target_initially_different=different,
            exact_restored_fields=fields,CPU_CUDA_python_numpy_next_draw_exact=True,order_view_next_draw_exact=True,
            pending_order_restored=True,next_logical_group=after['order'][:16],nonempty_zero_AdamW_slots_restored=len(after['optimizer']['state']),
            AdamW_slots_were_explicit_zero_update_fixtures=True,completed_epochs=0,TOP5_created=False,recipe_mismatch_rejected=True,
            recovered_initial_sha256=initial,checkpoint_sha256=digest(folder/'last.pt'),implementation_hash=frozen['implementation_hash']))
        print('Original complete-state recovery PASS; updates 0',flush=True)
if __name__=='__main__':main()
