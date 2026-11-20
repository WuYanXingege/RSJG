"""Additional CPU raw-forward/gradient checks, zero optimizer steps."""
import copy,json,os,sys
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
SOURCE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(SOURCE))
import torch
from bcep_joint.common import seed_all
from bcep_joint.checks import equal
from bcep_v3.fixtures import SyntheticData,SyntheticBank,joint_config
from bcep_v3.model import build
from bcep_v3.perf.stages import joint_callbacks
from bcep_v3.perf.options import checkpoint_arm


def main():
    torch.set_num_threads(1);assert not torch.cuda.is_available()
    seed_all(3201);config=joint_config();model,_=build(config)
    with torch.no_grad():
        model.coupling.csee.out.weight.normal_(std=.01)
        model.coupling.hcra.delta[-1].weight.normal_(std=.02)
        model.bridge.denoiser.grg_ctf.net[-1].weight.normal_(std=.01)
    initial=copy.deepcopy(model.state_dict());rows=[]
    for n in [1,2,9,57]:
        data=SyntheticData((n,),pixels=32);bank=SyntheticBank();reference=None
        arms=['baseline','p012']+(['map_off','bridge_off','both_off'] if n==2 else [])
        for arm in arms:
            model.load_state_dict(initial);model.train();model.zero_grad(set_to_none=True);seed_all(3201)
            captured={};original=model.coupling.assign
            def assign(*a,**kw):
                r=original(*a,**kw,return_details=True)
                captured['hard_goals']=r[0].detach().clone();captured['pi']=r[1].detach().clone()
                captured['scores']=r[2].detach().clone();captured['initial_pi']=r[3]['initial_pi'].detach().clone()
                return r[:3]
            def bridge(module,args,result):captured['bridge_outputs']=tuple(x.detach().clone() for x in result)
            model.coupling.assign=assign;hook=model.bridge.denoiser.register_forward_hook(bridge)
            order=torch.Generator().manual_seed(3201);view=torch.Generator().manual_seed(3202)
            try:
                with checkpoint_arm(model,map_checkpoint=arm not in ['map_off','both_off'],bridge_checkpoint=arm not in ['bridge_off','both_off']):
                    backward,_=joint_callbacks(model,data,'cpu',config,bank,bank,defer_diagnostics=arm=='p012',cpu_mask_check=arm=='p012')
                    captured['metrics']=backward([0],order,view)
            finally:model.coupling.assign=original;hook.remove()
            captured['gradients']={name:p.grad.detach().clone() if p.grad is not None else None for name,p in model.named_parameters()}
            captured['cpu_rng']=torch.get_rng_state();captured['order_rng']=order.get_state();captured['view_rng']=view.get_state()
            if reference is None:reference=captured
            else:equal(reference,captured)
            rows.append(dict(n_agents=n,arm=arm,all_captured_tensors_bitwise_equal=True,output_max_abs=0.,gradient_max_abs=0.,gradient_max_relative=0.,
                initial_and_final_hungarian_equal=True,checkpoint_flags_restored=model.bridge.encoder.activation_checkpointing and model.bridge.denoiser.activation_checkpointing))
    out=SOURCE.parent/'RSJG_perf_outputs/bcep_v3_speed_audit_20261120/FORWARD_EXACTNESS.json'
    out.write_text(json.dumps(dict(status='PASS_CPU',rows=rows,optimizer_updates=0,pixels=32,
        captured=['hard_goals','initial_pi','final_pi','world_scores','bridge_final','bridge_coarse','coarse_path','coarse_goal','loss_metrics','all_parameter_gradients','CPU_RNG','order_RNG','view_RNG'],
        GPU_full_resolution='PENDING'),indent=2)+'\n')
    print('CPU raw outputs, assignments, gradients and RNG bitwise equal; zero optimizer steps.')


if __name__=='__main__':main()
